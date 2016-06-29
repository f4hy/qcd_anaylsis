def all_equal(lst):
    return not lst or lst.count(lst[0]) == len(lst)
