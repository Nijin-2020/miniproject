n = int(input("How many elements? "))
my_list = []
for i in range(n):
    element = input(f"Enter element {i+1}: ")
    my_list.append(element)
print(my_list)
