import argparse


def argument_parse():
    parser = argparse.ArgumentParser(description='dnn conf')
    parser.add_argument(
        '--conf_path',
        dest='conf_path',
        default='./conf/settings/',
        help='dnn config file path')
    parser.add_argument(
        '--conf_file',
        dest='conf_file',
        default='demo.conf',
        help='dnn train config file, whose name would be the tag')
    parser.add_argument(
        '--model_ckpt',
        dest='model_ckpt',
        default='model.ckpt-0',
        help='dnn train ckpt file, whose name would be the global step')
    parser.add_argument(
        '--is_train',
        dest='is_train',
        default='false',
        help='if it is set to false, then train, or test')
    parser.add_argument(
        '--is_valid',
        dest='is_valid',
        default='false',
        help='if it is set to false, then train, or test')
    parser.add_argument(
        '--test_tag',
        dest='test_tag',
        default='clk',
        help='clk or ord')
    parser.add_argument(
        '--test_score_method',
        dest='test_score_method',
        default='ctr',
        help='rel or ctr')
    parser.add_argument(
        '--is_test',
        dest='is_test',
        default='false',
        help='if it is set to false, then train, or test')
    args = parser.parse_args()

    # vars() function return a dict type, return object's attributes and value of attributes
    return vars(args)


if __name__ == '__main__':
    args = argument_parse()
    # args is a dict
    print(args)
    # print args['conf_path']
    # print args['conf_file']
    for arg, value in args.items():
        print("{arg}: {value}".format(arg=arg, value=value))
