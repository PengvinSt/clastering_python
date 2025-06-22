import logging

class LogColors:
    RESET = "\033[0m"
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BRIGHT_BLACK = "\033[0;90m"
    BRIGHT_RED = "\033[0;91m"
    BRIGHT_GREEN = "\033[0;92m"
    BRIGHT_YELLOW = "\033[0;93m"
    BRIGHT_BLUE = "\033[0;94m"
    BRIGHT_MAGENTA = "\033[0;95m"
    BRIGHT_CYAN = "\033[0;96m"
    BRIGHT_WHITE = "\033[0;97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Styles
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    INVERSE = "\033[7m"
    STRIKETHROUGH = "\033[9m"

class ColoredFormatter(logging.Formatter):
    FORMAT = (
        f"{LogColors.BRIGHT_BLACK}%(asctime)s{LogColors.RESET} - "
        f"{LogColors.BLUE}[Process:{LogColors.CYAN}%(process)d{LogColors.BLUE}]{LogColors.RESET} - "
        f"{LogColors.BOLD}%(levelname)s{LogColors.RESET}: "
        f"{LogColors.WHITE}%(message)s{LogColors.RESET}"
    )

    FORMATS = {
        logging.DEBUG: (
            f"{LogColors.BRIGHT_BLACK}%(asctime)s{LogColors.RESET} - "
            f"{LogColors.BLUE}[Process:{LogColors.CYAN}%(process)d{LogColors.BLUE}]{LogColors.RESET} - "
            f"{LogColors.BOLD}{LogColors.BLUE}%(levelname)s{LogColors.RESET}: "
            f"{LogColors.WHITE}%(message)s{LogColors.RESET}"
        ),
        logging.INFO: (
            f"{LogColors.BRIGHT_BLACK}%(asctime)s{LogColors.RESET} - "
            f"{LogColors.BLUE}[Process:{LogColors.CYAN}%(process)d{LogColors.BLUE}]{LogColors.RESET} - "
            f"{LogColors.BOLD}{LogColors.GREEN}%(levelname)s{LogColors.RESET}: "
            f"{LogColors.WHITE}%(message)s{LogColors.RESET}"
        ),
        logging.WARNING: (
            f"{LogColors.BRIGHT_BLACK}%(asctime)s{LogColors.RESET} - "
            f"{LogColors.BLUE}[Process:{LogColors.CYAN}%(process)d{LogColors.BLUE}]{LogColors.RESET} - "
            f"{LogColors.BOLD}{LogColors.YELLOW}%(levelname)s{LogColors.RESET}: "
            f"{LogColors.WHITE}{LogColors.YELLOW}%(message)s{LogColors.RESET}" # Message also yellow
        ),
        logging.ERROR: (
            f"{LogColors.BRIGHT_BLACK}%(asctime)s{LogColors.RESET} - "
            f"{LogColors.BLUE}[Process:{LogColors.CYAN}%(process)d{LogColors.BLUE}]{LogColors.RESET} - "
            f"{LogColors.BOLD}{LogColors.RED}%(levelname)s{LogColors.RESET}: "
            f"{LogColors.WHITE}{LogColors.RED}%(message)s{LogColors.RESET}" # Message also red
        ),
        logging.CRITICAL: (
            f"{LogColors.BRIGHT_BLACK}%(asctime)s{LogColors.RESET} - "
            f"{LogColors.BLUE}[Process:{LogColors.CYAN}%(process)d{LogColors.BLUE}]{LogColors.RESET} - "
            f"{LogColors.BOLD}{LogColors.RED}{LogColors.BG_YELLOW}%(levelname)s{LogColors.RESET}: " # Critical with red text on yellow background
            f"{LogColors.WHITE}{LogColors.RED}%(message)s{LogColors.RESET}" # Message also red
        )
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)



def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    # Use the custom colored formatter
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger