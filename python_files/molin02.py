'''
@author: xiangfeng
@email: ample1666@gmail.com
@create_time: 2018/5/13 09:34
'''

'''
单链表的操作
'''
from node import Node, TwoWayNode

node1 = Node("a", None)
node2 = Node("b", None)
node3 = Node("c", None)
node4 = Node("d", None)
node5 = Node("e", None)

head = node1
node1.next = node2
node2.next = node3
node3.next = node4
node4.next = node5

#遍历单列表
# while head != None:
#     print(head.data)
#     head = head.next

#-----------------------------------------------

#搜索单列表
# while head != None and head.data != "g":
#     head = head.next
# if head != None:
#     print("find it")
# else:
#     print("not find it")

#-----------------------------------------------

#访问链表中的第i项
# index = 3
# while index > 0:
#     index -= 1
#     head = head.next
# print("the result is %s" % head.data)
#-----------------------------------------------

#在第i个位置插入
# index = 3
# while index > 2:
#     index -= 1
#     head = head.next
# head.next = Node("jassica", head.next)
#
# head = node1
# while head != None:
#     print(head.data)
#     head = head.next
#-----------------------------------------------

#删除第i个节点
# index = 4
# while index > 2:
#     index -= 1
#     head = head.next
# head.next = head.next.next
# head = node1
# while head != None:
#     print(head.data)
#     head = head.next

#-----------------------------------------------









