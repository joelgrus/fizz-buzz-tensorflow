import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-d", "--decimal", action="store_true", help="use decimal encoding (instead of binary)")
arg_parser.add_argument("--num_digits", type=int, help="number of digits")
arg_parser.add_argument("--batch_size", type=int, help="batch size")
arg_parser.add_argument("--num_epochs", type=int, help="epoch size")
arg_parser.add_argument("--num_hidden", type=int, help="number of first layer hidden units")
arg_parser.add_argument("--num_hidden2", type=int, help="number of second layer hidden units")
arg_parser.add_argument("--use_existing_model", action="store_true")
arg_parser.add_argument("--keep_prob", type=float, help="keep prob")
