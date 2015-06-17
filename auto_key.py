#!/usr/bin/env python2
from itertools import cycle
import logging

used_colors = {}

colorcycle = cycle('brcmkg')

markercycle = cycle('o<^>Dp8')
facecycle = cycle([None, "white"])


memoized_ids = {}

color_ids = {}
mark_ids = {}
face_ids = {}


def auto_key(identifier):
    print memoized_ids
    print color_ids, mark_ids, face_ids
    if identifier in memoized_ids:
        return memoized_ids[identifier]

    c = m = f = None
    try:
        new = list(enumerate(identifier))
        for i in enumerate(identifier):
            if i in color_ids:
                c = color_ids[i]
                new.remove(i)
                continue
            if i in mark_ids:
                m = mark_ids[i]
                new.remove(i)
                continue
            if i in face_ids:
                f = face_ids[i]
                new.remove(i)
                continue
    except TypeError:
        logging.info("not iterable!!")
        c = colorcycle.next()
        m = markercycle.next()
        f = c
        memoized_ids[identifier] = (c,m,f)
        return c,m,f

    print "new", new
    print face_ids
    if c is None:
        c = colorcycle.next()
        color_ids[new.pop(0)] = c
    if m is None and new:
        m = markercycle.next()
        mark_ids[new.pop(0)] = m
    if f is None and new:
        f = facecycle.next()
        if f == None:
            f = c
        face_ids[new.pop(0)] = f
    if f == None:
        f = c

    if (c,m,f) in memoized_ids.values():
        print "wtf", (c,m,f), "already memoized"
        print identifier
        exit(-1)

    memoized_ids[identifier] = (c,m,f)
    return c,m,f


if __name__ == "__main__":
    print("testing auto_key")

    print 5, 2, auto_key((5,2))
    print 5, 3, auto_key((5,3))
    print 5, 4, auto_key((5,4))
    print 5, 5, auto_key((5,5))
    print 5, 6, auto_key((5,6))
    print 4, 2, auto_key((4,4))
    print 4, 2, auto_key((4,5))
    exit(-1)
