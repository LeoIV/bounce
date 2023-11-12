class BColors:
    HEADER = "\033[95m"
    GREY = "\033[90m"
    WHITE = "\033[97m"
    LIGHTGREY = "\033[37m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


BOUNCE_NAME = """
    \t    ____                            
    \t   / __ )____  __  ______  ________
    \t  / __  / __ \/ / / / __ \/ ___/ _ \\
    \t / /_/ / /_/ / /_/ / / / / /__/  __/
    \t/_____/\____/\__,_/_/ /_/\___/\___/
    
    [ Bounce: a Reliable Bayesian Optimization     ] 
    [ Algorithm for Combinatorial and Mixed Spaces ]
    """

BOUNCE_NAME = f"{BColors.LIGHTGREY}{BOUNCE_NAME}{BColors.ENDC}"
