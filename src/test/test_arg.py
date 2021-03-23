import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group1 = parser.add_argument_group('group1')
    group1.add_argument('--test1', type=int, default=0, help="test1")

    group2 = parser.add_argument_group('group2')
    group2.add_argument('--test2', type=str, default="binh", help="test2")

    args = parser.parse_args()
    print("args:", args)
    print("group1:", group1)
    print("group2:", group2)
    print("parser._action_groups:", parser._action_groups)
    print([group.title for group in parser._action_groups])

    for group in parser._action_groups:
        print("group:", group)
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        print(group.title, argparse.Namespace(**group_dict))

    group_dict = {a.dest: getattr(args, a.dest, None) for a in group1._group_actions}
    group1_args = argparse.Namespace(**group_dict)
    print("group1_args:", group1_args.test1)
