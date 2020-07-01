def plus():
    g_sum = int(input())
    a = 0
    while True:
        b = g_sum - a
        (a+b) == g_sum
        print(a, b)
        if a > b:
            break
        a += 1

