import logging
import logging.handlers
from colorama import Fore, Style, init

# 初始化Colorama库
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Style.BRIGHT + Fore.BLUE,
        'INFO': Style.BRIGHT + Fore.GREEN,
        'WARNING': Style.BRIGHT + Fore.YELLOW,
        'ERROR': Style.BRIGHT + Fore.RED,
        'CRITICAL': Style.BRIGHT + Fore.RED
    }

    def format(self, record: logging.LogRecord):
        log_message = super().format(record)
        return self.COLORS.get(record.levelname, '') + log_message + Style.RESET_ALL
    
def is_logging_enabled():
    from sglang.srt.layers.dp_attention import (
        is_dp_attention_enabled,
        get_attention_dp_rank,
        get_attention_tp_rank
    )
    return get_attention_tp_rank() == 0
    if is_dp_attention_enabled():
        return get_attention_dp_rank() == 0
    else:
        return get_attention_tp_rank() == 0


if __name__ == "__main__":
    # 创建Logger实例
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # logger.propagate = False

    # 创建控制台处理器，并设置输出格式
    console_formatter = ColoredFormatter('[%(asctime)s] %(levelname)8s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    # logging.basicConfig(handlers=[console_handler], force=True)
    logging.basicConfig(handlers=[console_handler])
    
    # logger.addHandler(console_handler)
    # print(logger.handlers)

    # 示例日志输出
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")
    logger.critical("这是一条严重错误")