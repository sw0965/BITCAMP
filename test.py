for i in [1, 2, 3, 4, 5]:
    print('only i :', i)
    for j in [1, 2, 3, 4, 5]:
        print('only j :', j)
        print('i+j: ',i+j)
    print(i)

def my_print(message = "hi"):
    print(message)

my_print("hello")
my_print()

first_name = "Han"
last_name = "Sangwoo"
full_name1 = first_name + ""+last_name
print(full_name1)

interger_list = [1,2,3]
heter = ["string", 0.1, True]
l_o_l = [interger_list, heter, []]
print(l_o_l)

x, y, z = [1, 2, 'hi']
print(x, y, z)