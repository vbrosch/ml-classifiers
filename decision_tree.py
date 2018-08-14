"""
    A simple decision tree
"""
import math

ident = '\t'

class Node:
    def __init__(self, subset: list, remaining_features: list):
        self.f_i = -1
        self.on0 = None
        self.on1 = None
        self.subset = subset
        self.remaining_features = remaining_features

    def classify(self, x):
        val = x[self.f_i]

        if val == 0:
            return self.on0.classify(x) if isinstance(self.on0, Node) else self.on0
        elif val == 1:
            return self.on1.classify(x) if isinstance(self.on1, Node) else self.on1

    def print(self, ident_prefix: str = ''):
        print(ident_prefix, end='')
        print('Node, split on f{}'.format(self.f_i))
        next_ident = ident_prefix + ident

        print('{} 0'.format(ident_prefix))
        if isinstance(self.on0, Node):
            self.on0.print(next_ident)
        else:
            print('{}-> C{}'.format(next_ident, self.on0))

        print('{} 1'.format(ident_prefix))
        if isinstance(self.on1, Node):
            self.on1.print(next_ident)
        else:
            print('{}-> C{}'.format(next_ident, self.on1))


def log_value(c_i):
    return c_i * math.log2(c_i) if c_i != 0 else 0


def class_probability(subset: list, f_i, f_v, c):
    norm = sum([1 for x in subset if x[0][f_i] == f_v])
    return sum([1 for x in subset if x[0][f_i] == f_v and x[1] == c]) / norm


def entropy_impurity(subset: list, f_i, f_v):
    p_c_0 = class_probability(subset, f_i, f_v, 0)
    p_c_1 = 1 - p_c_0

    return -(log_value(p_c_0) + log_value(p_c_1))


def subtree_entropy_impurity(subset: list, f_i):
    norm = sum([1 for _ in subset])
    p_f_0 = sum([1 for x in subset if x[0][f_i] == 0]) / norm
    p_f_1 = sum([1 for x in subset if x[0][f_i] == 1]) / norm

    return p_f_0 * entropy_impurity(subset, f_i, 0) + p_f_1 * entropy_impurity(subset, f_i, 1)


def best_split_feature(node: Node) -> int:
    i = -1
    min_impurity = 2

    for f_i in node.remaining_features:
        impurity = subtree_entropy_impurity(node.subset, f_i)

        if impurity < min_impurity:
            i = f_i
            min_impurity = impurity

    print("Will split at feature {} with impurity {}".format(i, min_impurity))

    return i


def remove_from_subset(node: Node, f_v):
    deletions = [n for n in node.subset if n[0][node.f_i] == f_v]

    for deletion in deletions:
        node.subset.remove(deletion)


def get_leave_class(node: Node, f_v) -> int:
    c0_prob = class_probability(node.subset, node.f_i, f_v, 0)
    return 0 if c0_prob == 1 else 1


def assign_leaves(node: Node):
    im_0 = math.fabs(entropy_impurity(node.subset, node.f_i, 0))
    im_1 = math.fabs(entropy_impurity(node.subset, node.f_i, 1))
    remaining_features = [x for x in node.remaining_features if x != node.f_i]

    if im_0 == 0:
        node.on0 = get_leave_class(node, 0)
        remove_from_subset(node, 0)
    else:
        subset_0 = [x for x in node.subset if x[0][node.f_i] == 0]
        node.on0 = Node(subset_0, remaining_features)
        node.on0 = train(node.on0)

    if im_1 == 0:
        node.on1 = get_leave_class(node, 1)
        remove_from_subset(node, 1)
    else:
        subset_1 = [x for x in node.subset if x[0][node.f_i] == 1]
        node.on1 = Node(subset_1, remaining_features)
        node.on1 = train(node.on1)


def can_assign_leaves(node: Node):
    return entropy_impurity(node.subset, node.f_i, 0) == 0 or entropy_impurity(node.subset, node.f_i, 1) == 0


def train(node: Node) -> Node:
    node.f_i = best_split_feature(node)
    assign_leaves(node)

    return node


c0 = [[0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]]
c1 = [[1, 0, 1, 1], [0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0]]

samples = [(x, 0) for x in c0] + [(x, 1) for x in c1]
unassigned_samples = samples.copy()

tree = train(Node(samples, list(range(4))))

print("\n### Will print learned decision tree now: ###\n")
tree.print()

s1 = [0, 0, 1, 1]
s2 = [1, 0, 1, 1]
s3 = [1, 1, 1, 0]

print("\nWill classify three samples: s1={}, s2={} and s3={}\n".format(s1, s2, s3))

result = tree.classify(s1)

print("{} -> C{}".format(s1, result))

result = tree.classify(s2)

print("{} -> C{}".format(s2, result))

result = tree.classify(s3)

print("{} -> C{}".format(s3, result))

