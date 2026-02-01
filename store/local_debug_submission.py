import sys
class ListNode:
	def __init__(self, item):
		self.item = item
		self.next = None

class LinkedList:
	def __init__(self):
		self.head = None
		self.size = 0

	def print_list(self):
		cur = self.head
		while cur.next:
			print(cur.item, end=",")
			cur = cur.next
		print(cur.item)
	
	def remove_all_items(self):
		self.head = None
		self.size = 0

	def find_node(self, index):
		if index < 0 or index >= self.size:
			return None
		cur = self.head
		while index > 0:
			cur = cur.next
			index -= 1
		return cur

	def insert_node(self, index, value):
		new_node = ListNode(value)
		if index == 0:
			new_node.next = self.head
			self.head = new_node
		else:
			prev = self.find_node(index - 1)
			if prev:
				new_node.next = prev.next
				prev.next = new_node
		self.size += 1

	def remove_node(self, index):
		if index < 0 or index >= self.size:
			return -1

		if index == 0:  # Remove the first node
			self.head = self.head.next
		else:
			prev = self.find_node(index - 1)
			prev.next = prev.next.next

		self.size -= 1
		return 0

# YOUR CODE HERE
def remove_duplicates_sorted_ll(ll):
	# If the list is empty or has only one node, nothing to do
	if ll.head is None or ll.head.next is None:
		return

	cur = ll.head
	# Traverse the list and remove consecutive duplicates
	while cur is not None and cur.next is not None:
		if cur.item == cur.next.item:
			# Skip the next node (duplicate)
			cur.next = cur.next.next
			ll.size -= 1
		else:
			# Move to the next distinct element
			cur = cur.next

	

def run_code(inputs):
	for input in inputs:
		input_list = input.split(",")
		# Initialize linked list
		ll = LinkedList()
		for num in input_list:
			ll.insert_node(ll.size, int(num))

		remove_duplicates_sorted_ll(ll)

		# Printing solution
		ll.print_list()

# An example on how to call this function:
# python <file_name>.py 1,1,2,3,3,3 --> printed output = 1,2,3
def main():
	input_list = sys.argv[1].split(",")
	# Initialize linked list
	ll = LinkedList()
	for num in input_list:
		ll.insert_node(ll.size, int(num))

	remove_duplicates_sorted_ll(ll)

	# Printing solution
	ll.print_list()

if __name__ == "__main__":
	main()