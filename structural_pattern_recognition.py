"""
    some sample implementations for structural pattern recognition
"""
import sys

from lina.linear_algebra import init_matrix

costs = [
    0, # a      -> a
    1, # a      -> eps
    1, # eps    -> a
    2  # a      -> b
]


def string_edit_distance(s1: str, s2: str, c: list) -> int:
    cost_matrix = init_matrix(nrows=len(s1)+1, ncols=len(s2)+1)

    # fill first row
    for i in range(1, len(s1)+1):
        cost_matrix[i][0] = cost_matrix[i-1][0] + c[1]

    # fill first col
    for j in range(1, len(s2)+1):
        cost_matrix[0][j] = cost_matrix[0][j-1] + c[2]

    for i in range(1, len(s1)+1):
        for j in range(1, len(s2)+1):

            unchanged_distance = sys.maxsize
            if s1[i-1] == s2[j-1]:
                unchanged_distance = cost_matrix[i-1][j-1] + c[0]

            change_distance = cost_matrix[i-1][j-1] + c[3]
            remove_distance = cost_matrix[i-1][j] + c[1]
            add_distance = cost_matrix[i][j-1] + c[2]

            cost_matrix[i][j] = min(change_distance, remove_distance, add_distance, unchanged_distance)

    print(cost_matrix)

    return cost_matrix[len(s1)][len(s2)]


x = 'INDUSTRY'
y = 'INTEREST'

distance = string_edit_distance(x, y, costs)

print("d({},{}) = {}".format(x, y, distance))
