'''
@author: xiangfeng
@email: ample1666@gmail.com
@create_time: 2018/5/8 21:33
'''

# r = float(input("输入球体的半径: "))
# print("球体的直径：%f" % (r*2)) #一定要加括号

#--------------------------------------

def indexOfMin(l):
    '''
    :param l: 一个列表
    :return: 列表中最小数的索引
    '''
    if len(l) == 0:
        return None
    if len(l) == 1:
        return 0

    minIndex = 0
    currentIndex = 1
    while currentIndex < len(l):
        if l[minIndex] > l[currentIndex]:
            minIndex = currentIndex
        currentIndex += 1
    return minIndex

#----------------------------

def linearSearch(target, l):
    '''
    顺序搜索列表中的目标项
    :param target: 目标项
    :param l: 一个列表
    :return: 目标项的索引，如果没有，则返回-1
    '''
    currentIndex = 0
    while currentIndex < len(l):
        if target == l[currentIndex]:
            return currentIndex
        else:
            currentIndex += 1
    return -1

#-----------------------------------

def binarySearch(target, sortedList):
    '''
    二分查找法
    :param target:
    :param sortedList:
    :return: 目标项的索引
    '''
    left = 0
    right = len(sortedList) - 1
    while left <= right:
        mid = (left + right) // 2
        print("the left: %d, the right: %d, the mid: %d" % (left, right, mid))
        if target == sortedList[mid]:
            return mid
        elif target < sortedList[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return -1

#-----------------------------------

def interpolationSearch(target, sortedList):
    '''
    插值查找法，适用于分布均匀的查找表，整体上与二分查找很像
    :param target:
    :param sortedList:
    :return:目标项的索引
    '''
    left = 0
    right = len(sortedList) - 1
    while left <= right:
        # 下面一行就是与二分查找唯一的区别了
        mid = left + (right - left) * (target - sortedList[left]) / (sortedList[right] - sortedList[left])
        mid = int(mid)
        print("the left: %d, the right: %d, the mid: %d" % (left, right, mid))
        if target == sortedList[mid]:
            return mid
        elif target < sortedList[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return -1

#-------------------------------------

def fibonacci(x):
    if x < 3:
        return 1
    return fibonacci(x-1) + fibonacci(x-2)

def fibonacciSearch(target, sortedList):
    # 生成fibonacci数列
    l = []
    for i in range(1, 20):
        l.append(fibonacci(i))

    length = len(sortedList)
    low, high = 0, length-1
    k = 0
    while length > l[k]-1:
        k += 1

    for i in range(length, l[k]-1):
        l.append(sortedList[length-1])

    while low <= high:
        mid = low + l[k-1] - 1
        if target > sortedList[mid]:
            low = mid + 1
            k -= 2
        elif target < sortedList[mid]:
            high = mid - 1
            k -= 1
        else:
            if mid <= length-1:
                return mid
            else:
                return length-1

    return -1

#-------------------------------------

def swap(l, i, j):
    '''
    在列表中交换两个元素的位置
    :param l: 一个列表
    :param i: 被交换的元素索引
    :param j: 被交换的元素索引
    :return:
    '''
    # temp = l[i]
    # l[i] = l[j]
    # l[j] = temp
    l[i], l[j] = l[j], l[i]

#-------------------------------------

def selectionSort(l):
    '''
    选择排序法，时间复杂度是O(n*n)
    :param l:
    :return:
    '''
    i = 0
    length = len(l)
    while i < length - 1:
        minIndex = i
        j = i + 1
        while j < length:
            if l[j] < l[minIndex]:
                minIndex = j
            j += 1
        if minIndex != i:
            swap(l, minIndex, i)
        i += 1

#-------------------------------------

def bubbleSort(l):
    '''
    冒泡排序法, 时间复杂度是O(n*n)
    :param l:
    :return:
    '''
    length = len(l)
    i = 0
    while i < length - 1:
        j = length - 1
        while j > i:
            if l[j-1] > l[j]:
                swap(l, j-1, j)
            j -= 1
        i += 1

#-------------------------------------

def frank_bubbleSort(l):
    '''
    对上面的冒泡排序进行了优化
    :param l:
    :return:
    '''
    length = len(l)
    i = 0
    flag = True
    while (i < length-1) and flag:
        flag = False
        j = length - 1
        while j > i:
            if l[j - 1] > l[j]:
                swap(l, j - 1, j)
                flag = True
            j -= 1
        i += 1

#-------------------------------------

def insertionSort(l):
    '''
    插入排序法，时间复杂度是O(n*n)
    :param l:
    :return:
    '''
    length = len(l)
    i = 1
    while i < length:
        tmp = l[i]
        j = i
        while (j > 0) and (tmp < l[j - 1]):
            l[j] = l[j - 1]
            j -= 1
        l[j] = tmp
        i += 1

#-------------------------------------

def shellSort(l):
    '''
    希尔排序
    :param l:
    :return:
    '''
    increment = len(l)
    while increment > 1:
        increment = increment // 3 + 1
        i = increment + 1
        while i <= len(l):
            if l[i] < l[i-increment]:
                l[0] = l[i]


#-------------------------------------

l = [88, 9,1,5,8,33, 3,12, 54, 7,4,6,2]

# print(fibonacciSearch(99, l))

insertionSort(l)
print(l)

