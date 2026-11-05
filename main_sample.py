from args import parse_args
from main import main

if __name__ == "__main__":
    args = parse_args()
    args.test = True
    main(args)
