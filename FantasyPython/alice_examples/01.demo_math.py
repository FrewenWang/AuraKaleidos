

def cal_e(n):
    sum:float = 0.0
    for i in range(n):
        sum += float(1 /( (i + 1) * (i + 1)))
    return sum




print(cal_e(100000000))