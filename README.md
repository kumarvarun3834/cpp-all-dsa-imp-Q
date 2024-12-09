# cpp all dsa imp Q
 
**Prompt:** Solve the given array and string problems with code explanations.

**Here are the solutions to the given problems, along with code implementations and explanations:**

**1. Find the maximum subarray sum (Kadane’s Algorithm):**

```python
def max_subarray_sum(nums):
    max_so_far = max_ending_here = nums[0]
    for num in nums[1:]:
        max_ending_here = max(num, max_ending_here + num)
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far
```

**Explanation:**
- **max_so_far:** Stores the maximum sum encountered so far.
- **max_ending_here:** Stores the maximum sum of the subarray ending at the current index.
- Iterates through the array, updating `max_ending_here` by comparing the current element with the sum of the current element and the previous `max_ending_here`.
- Updates `max_so_far` by comparing it with the current `max_ending_here`.

**2. Find duplicates in an array of integers:**

```python
def find_duplicates(nums):
    seen = set()
    duplicates = set()
    for num in nums:
        if num in seen:
            duplicates.add(num)
        seen.add(num)
    return list(duplicates)
```

**Explanation:**
- Uses a set to keep track of seen numbers.
- If a number is encountered again, it's added to the `duplicates` set.

**3. Merge two sorted arrays without using extra space:**

```python
def merge_sorted_arrays(nums1, m, nums2, n):
    p1 = m - 1
    p2 = n - 1
    i = m + n - 1
    while p2 >= 0:
        if p1 >= 0 and nums1[p1] > nums2[p2]:
            nums1[i] = nums1[p1]
            p1 -= 1
        else:
            nums1[i] = nums2[p2]
            p2 -= 1
        i -= 1
```

**Explanation:**
- Starts from the end of both arrays and iteratively compares elements.
- Places the larger element at the end of `nums1`.

**4. Find the first non-repeating character in a string:**

```python
def first_non_repeating_character(s):
    char_counts = {}
    for char in s:
        char_counts[char] = char_counts.get(char, 0) + 1
    for char in s:
        if char_counts[char] == 1:
            return char
    return None
```

**Explanation:**
- Creates a dictionary to count the frequency of each character.
- Iterates through the string, returning the first character with a count of 1.

**5. Rotate an array by k positions:**

```python
def rotate_array(nums, k):
    k %= len(nums)
    reverse(nums, 0, len(nums) - 1)
    reverse(nums, 0, k - 1)
    reverse(nums, k, len(nums) - 1)

def reverse(nums, start, end):
    while start < end:
        nums[start], nums[end] = nums[end], nums[start]
        start += 1
        end -= 1
```

**Explanation:**
- Reverses the entire array, then reverses the first `k` and last `n-k` elements.

**6. Find the majority element in an array (element appearing > n/2 times):**

```python
def majority_element(nums):
    candidate, count = None, 0
    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1
    return candidate
```

**Explanation:**
- Uses Boyer-Moore Voting Algorithm to find the majority element.

**7. Search for an element in a rotated sorted array:**

```python
def search_in_rotated_sorted_array(nums, target):
    low, high = 0, len(nums) - 1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] >= nums[low]:  # Left half is sorted
            if target >= nums[low] and target < nums[mid]:
                high = mid - 1
            else:
                low = mid + 1
        else:  # Right half is sorted
            if target > nums[mid] and target <= nums[high]:
                low = mid + 1
            else:
                high = mid - 1
    return -1
```

**Explanation:**
- Finds the pivot point to determine the sorted halves.
- Uses binary search on the appropriate half.

**8. Find the longest substring without repeating characters:**

```python
def length_of_longest_substring(s):
    char_index_map = {}
    left = 0
    max_len = 0
    for right in range(len(s)):
        if s[right] in char_index_map:
            left = max(left, char_index_map[s[right]] + 1)
        char_index_map[s[right]] = right
        max_len = max(max_len, right - left + 1)
    return max_len
```

**Explanation:**
- Uses a sliding window approach with a hash map to keep track of character indices.
- Expands the window as long as characters are unique.
- Shrinks the window when a duplicate character is encountered.

**9. Two-pointer techniques:**

* **Pair sum in a sorted array:**
  ```python
  def pair_sum(nums, target):
      left, right = 0, len(nums) - 1
      while left < right:
          current_sum = nums[left] + nums[right]
          if current_sum == target:
              return [nums[left], nums[right]]
          elif current_sum < target:
              left += 1
          else:
              right -= 1
      return []
  ```
* **Three-sum or four-sum problems:**
  Similar to the two-pointer approach for pair sum, but with additional loops and pointers to handle multiple sums.

**10. Palindrome and permutation palindrome:**

* **Palindrome:**
  ```python
  def is_palindrome(s):
      return s == s[::-1]
  ```
* **Permutation palindrome:**
  ```python
  def is_permutation_palindrome(s):
      char_counts = {}
      for char in s:
          char_counts[char] = char_counts.get(char, 0) + 1
      odd_count = 0
      for count in char_counts.values():
          odd_count += count % 2
      return odd_count <= 1
  ```

**Remember to adapt these solutions to specific problem constraints and edge cases.**

**Stacks and Queues**

**1. Implement a stack using arrays or linked lists:**

**Array-based implementation:**

```python
class Stack:
    def __init__(self):
        self.stack = []
        self.top = -1

    def push(self, item):
        self.top += 1
        self.stack.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        self.top -= 1
        return self.stack.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.stack[self.top]

    def is_empty(self):
        return self.top == -1
```

**Linked list-based implementation:**

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Stack:
    def __init__(self):
        self.head = None

    def push(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node

    def pop(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        temp = self.head
        self.head = self.head.next
        return temp.data

    def peek(self):
        if self.is_empty():
            raise IndexError("Stack is empty")
        return self.head.data

    def is_empty(self):
        return self.head is None
```

**2. Implement a queue using stacks:**

```python
class Queue:
    def __init__(self):
        self.stack1 = []  # For enqueue
        self.stack2 = []  # For dequeue

    def enqueue(self, item):
        self.stack1.append(item)

    def dequeue(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        if not self.stack2:
            raise IndexError("Queue is empty")
        return self.stack2.pop()
```

**3. Design a min-stack:**

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]
```

**4. Evaluate a postfix expression:**

```python
def evaluate_postfix(expression):
    stack = []
    for char in expression:
        if char.isdigit():
            stack.append(int(char))
        else:
            val2 = stack.pop()
            val1 = stack.pop()
            if char == '+':
                stack.append(val1 + val2)
            elif char == '-':
                stack.append(val1 - val2)
            elif char == '*':
                stack.append(val1 * val2)
            elif char == '/':
                stack.append(int(val1 / val2))
    return stack.pop()
```

**5. Implement a circular queue:**

```python
class CircularQueue:
    def __init__(self, capacity):
        self.queue = [None] * capacity
        self.capacity = capacity
        self.front = self.rear = -1

    def enqueue(self, data):
        if (self.rear + 1) % self.capacity == self.front:
            print("Queue is full")
            return
        if self.front == -1:
            self.front = 0
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = data

    def dequeue(self):
        if self.front == -1:
            print("Queue is empty")
            return
        if self.front == self.rear:
            temp = self.queue[self.front]
            self.front = self.rear = -1
            return temp
        temp = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        return temp
```

**6. Check for balanced parentheses in an expression:**

```python
def is_balanced(expression):
    stack = []
    for char in expression:
        if char in ['(', '{', '[']:
            stack.append(char)
        elif char in [')', '}', ']']:
            if not stack:
                return False
            top_element = stack.pop()
            if (char == ')' and top_element != '(') or (char == '}' and top_element != '{') or (char == ']' and top_element != '['):
                return False
    return not stack
```

**7. Find the largest rectangular area in a histogram:**

```python
def largestRectangleArea(heights):
    stack = []
    max_area = 0
    i = 0
    while i < len(heights):
        if not stack or heights[stack[-1]] <= heights[i]:
            stack.append(i)
            i += 1
        else:
            top_index = stack.pop()
            area = heights[top_index] * (i - stack[-1] - 1 if stack else i)
            max_area = max(max_area, area)
    while stack:
        top_index = stack.pop()
        area = heights[top_index] * (i - stack[-1] - 1 if stack else i)
        max_area = max(max_area, area)
    return max_area
```

**8. Sliding window maximum using deque or monotonic queue:**

```python
from collections import deque

def maxSlidingWindow(nums, k):
    q = deque()
    result = []
    for i in range(len(nums)):
        while q and nums[i] >= nums[q[-1]]:
            q.pop()
        q.append(i)
        if i - q[0] >= k:
            q.popleft()
        if i >= k - 1:
            result.append(nums[q[0]])
    return result
```

**Note:** These are basic implementations and can be optimized further. Consider the specific requirements of your application and explore additional techniques like lazy propagation for more complex scenarios.

**Linked Lists**

**1. Reverse a Linked List:**

**Iterative Approach:**

```python
def reverseList(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

**Recursive Approach:**

```python
def reverseListRecursive(head):
    if not head or not head.next:
        return head
    p = reverseListRecursive(head.next)
    head.next.next = head
    head.next = None
    return p
```

**2. Detect a Cycle in a Linked List (Floyd's Cycle-Finding Algorithm):**

```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

**3. Merge Two Sorted Linked Lists:**

```python
def mergeTwoLists(list1, list2):
    dummy = ListNode()
    tail = dummy
    while list1 and list2:
        if list1.val < list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    if list1:
        tail.next = list1
    elif list2:
        tail.next = list2
    return dummy.next
```

**4. Find the Middle of a Linked List in One Pass:**

```python
def middleNode(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

**5. Remove n-th Node from the End of a Linked List:**

```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0)
    dummy.next = head
    first = second = dummy
    for i in range(n + 1):
        first = first.next
    while first:
        first = first.next
        second = second.next
    second.next = second.next.next
    return dummy.next
```

**6. Flatten a Multilevel Doubly Linked List:**

```python
def flatten(head):
    if not head:
        return head
    stack = [head]
    prev = None
    while stack:
        curr = stack.pop()
        curr.prev = prev
        if prev:
            prev.next = curr
        prev = curr
        if curr.next:
            stack.append(curr.next)
        if curr.child:
            stack.append(curr.child)
            curr.child = None
    head.prev = None
    return head
```

**7. Add Two Numbers Represented by Linked Lists:**

```python
def addTwoNumbers(l1, l2):
    dummyHead = ListNode(0)
    curr = dummyHead
    carry = 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        carry, val = divmod(val1 + val2 + carry, 10)
        curr.next = ListNode(val)
        curr = curr.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None
    return dummyHead.next
```

**8. Check if Two Linked Lists Intersect:**

```python
def getIntersectionNode(headA, headB):
    if not headA or not headB:
        return None
    pa = headA
    pb = headB
    while pa != pb:
        pa = pa.next if pa else headB
        pb = pb.next if pb else headA
    return pa
```

**1. Tree Traversals**

**In-order Traversal:**
* Visit left subtree
* Visit root node
* Visit right subtree

```python
def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)
        print(root.val)
        inorder_traversal(root.right)
```

**Pre-order Traversal:**
* Visit root node
* Visit left subtree
* Visit right subtree

```python
def preorder_traversal(root):
    if root:
        print(root.val)
        preorder_traversal(root.left)
        preorder_traversal(root.right)
```

**Post-order Traversal:**
* Visit left subtree
* Visit right subtree
* Visit root node

```python
def postorder_traversal(root):
    if root:
        postorder_traversal(root.left)
        postorder_traversal(root.right)
        print(root.val)
```

**Level-order Traversal:**
* Visit nodes level by level, from left to right

```python
def level_order_traversal(root):
    if not root:
        return []
    queue = [root]
    result = []
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            node = queue.pop(0)
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(current_level)
    return result
```

**2. Height of a Binary Tree**

```python
def height(root):
    if not root:
        return 0
    return 1 + max(height(root.left), height(root.right))
```

**3. Check if a Binary Tree is Balanced**

```python
def is_balanced(root):
    if not root:
        return True
    left_height = height(root.left)
    right_height = height(root.right)
    return abs(left_height - right_height) <= 1 and is_balanced(root.left) and is_balanced(root.right)
```

**4. Lowest Common Ancestor (LCA) of Two Nodes in a Binary Tree**

```python
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left or right
```

**5. Serialize and Deserialize a Binary Tree**

```python
def serialize(root):
    if not root:
        return "null,"
    return str(root.val) + "," + serialize(root.left) + serialize(root.right)

def deserialize(data):
    def helper():
        val = next(vals)
        if val == "null":
            return None
        node = TreeNode(int(val))
        node.left = helper()
        node.right = helper()
        return node
    vals = iter(data.split(","))
    return helper()
```

**6. Construct a Binary Tree from Inorder and Preorder/Postorder Traversal**

```python
def build_tree_from_inorder_preorder(inorder, preorder):
    if not inorder or not preorder:
        return None
    root_val = preorder[0]
    root_index = inorder.index(root_val)
    root = TreeNode(root_val)
    root.left = build_tree_from_inorder_preorder(inorder[:root_index], preorder[1:root_index+1])
    root.right = build_tree_from_inorder_preorder(inorder[root_index+1:], preorder[root_index+1:])
    return root
```

**7. Validate if a Binary Tree is a Binary Search Tree (BST)**

```python
def is_bst(root, min_val=float('-infinity'), max_val=float('infinity')):
    if not root:
        return True
    if root.val < min_val or root.val > max_val:
        return False
    return is_bst(root.left, min_val, root.val) and is_bst(root.right, root.val, max_val)
```

**8. Find the Diameter of a Binary Tree**

```python
def diameter_of_binary_tree(root):
    def dfs(root):
        if not root:
            return 0, 0
        left_height, left_diameter = dfs(root.left)
        right_height, right_diameter = dfs(root.right)
        diameter = max(left_height + right_height + 1, left_diameter, right_diameter)
        height = 1 + max(left_height, right_height)
        return height, diameter
    _, diameter = dfs(root)
    return diameter
```

**9. Flatten a Binary Tree to a Linked List**

```python
def flatten(root):
    if not root:
        return None
    prev = None
    stack = [root]
    while stack:
        node = stack.pop()
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
        node.right = prev
        node.left = None
        prev = node
    return root
```

**10. Find the kth Smallest/Largest Element in a BST**

```python
def kth_smallest(root, k):
    def inorder_traversal(node, k, count):
        if not node:
            return None
        left = inorder_traversal(node.left, k, count)
        if left:
            return left
        count[0] += 1
        if count[0] == k:
            return node
        return inorder_traversal(node.right, k, count)
    count = [0]
    return inorder_traversal(root, k, count)
```


**1. Depth First Search (DFS)**

```python
def dfs_recursive(node, visited):
    visited.add(node)
    print(node, end=" ")
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(neighbor, visited)

def dfs_iterative(start_node):
    visited = set()
    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            print(node, end=" ")
            for neighbor in graph[node]:
                stack.append(neighbor)
```

**2. Breadth First Search (BFS)**

```python
def bfs(start_node):
    visited = set()
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            print(node, end=" ")
            for neighbor in graph[node]:
                queue.append(neighbor)
```

**3. Check if a Graph is Bipartite**

```python
def is_bipartite(graph):
    colors = {}
    for node in graph:
        if node not in colors:
            if not dfs_bipartite(node, colors, 0):
                return False
    return True

def dfs_bipartite(node, colors, color):
    colors[node] = color
    for neighbor in graph[node]:
        if neighbor not in colors:
            if not dfs_bipartite(neighbor, colors, 1 - color):
                return False
        elif colors[neighbor] == color:
            return False
    return True
```

**4. Shortest Path in an Unweighted Graph (BFS)**

```python
def shortest_path(graph, start_node, end_node):
    queue = [(start_node, 0)]
    visited = set()
    while queue:
        node, distance = queue.pop(0)
        if node == end_node:
            return distance
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
    return -1
```

**5. Dijkstra's Algorithm for Shortest Path in a Weighted Graph**

```python
import heapq

def dijkstra(graph, start_node):
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    heap = [(0, start_node)]

    while heap:
        current_distance, current_node = heapq.heappop(heap)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    return distances
```

**6. Detect a Cycle in a Graph (Directed/Undirected)**

```python
def is_cyclic_undirected(graph):
    visited = set()
    parent = {}
    for node in graph:
        if node not in visited:
            if dfs_cycle_undirected(node, -1, visited, parent):
                return True
    return False

def dfs_cycle_undirected(node, parent, visited, parent_dict):
    visited.add(node)
    parent_dict[node] = parent
    for neighbor in graph[node]:
        if neighbor not in visited:
            if dfs_cycle_undirected(neighbor, node, visited, parent_dict):
                return True
        elif neighbor != parent:
            return True
    return False

def is_cyclic_directed(graph, node, visited, rec_stack):
    visited[node] = True
    rec_stack[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            if is_cyclic_directed(graph, neighbor, visited, rec_stack):
                return True
        elif rec_stack[neighbor]:
            return True
    rec_stack[node] = False
    return False
```

**7. Find Connected Components in a Graph**

```python
def connected_components(graph):
    visited = set()
    count = 0
    for node in graph:
        if node not in visited:
            dfs_connected_components(node, visited)
            count += 1
    return count

def dfs_connected_components(node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_connected_components(neighbor, visited)
```

**8. Kruskal's Algorithm for Minimum Spanning Tree (MST)**

```python
def kruskal_mst(graph):
    edges = []
    for node in graph:
        for neighbor, weight in graph[node].items():
            edges.append((weight, node, neighbor))
    edges.sort()
    parent = {node: node for node in graph}
    mst = []
    for weight, u, v in edges:
        if find_parent(parent, u) != find_parent(parent, v):
            mst.append((u, v, weight))
            union(parent, u, v)
    return mst

def find_parent(parent, node):
    if parent[node] == node:
        return node
    return find_parent(parent, parent[node])

def union(parent, x, y):
    parent_x = find_parent(parent, x)
    parent_y = find_parent(parent, y)
    parent[parent_x] = parent_y
```

**9. Solve a Maze or Grid Problem using BFS/DFS**

```python
def solve_maze(maze, start, end):
    queue = [start]
    visited = set()
    while queue:
        x, y = queue.pop(0)
        if (x, y) == end:
            return True
        if (x, y) not in visited and maze[x][y] == 1:
            visited.add((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < len(maze) and 0 <= new_y < len(maze[0]):
                    queue.append((new_x, new_y))
    return False
```

**10. Topological Sorting of a Directed Acyclic Graph (DAG)**

```python
def topological_sort(graph):
    indegree = {node: 0 for node in graph}
    for node in graph:
        for neighbor in graph[node]:
            indegree[neighbor] += 1
    queue = [node for node in indegree if indegree[node] == 0]
    topological_order = []
    while queue:
        node = queue.pop(0)
        topological_order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    return topological_order
```

**11. Word Ladder Problem**

```python
from collections import deque

def word_ladder(beginWord, endWord, wordList):
    wordList = set(wordList)
    queue = deque([(beginWord, 1)])
    visited = set([beginWord])
    while queue:
        word, level = queue.popleft()
        if word == endWord:
            return level
        for i in range(len(word)):
            for char in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + char + word[i+1:]
                if new_word in wordList and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, level + 1))
    return 0
```

These are some of the common graph algorithms and their implementations. Remember to adapt the code to your specific use case and data structures.


**N-Queens Problem**

```python
def solve_n_queens(n):
    def solve(board, col):
        if col >= n:
            return True

        for row in range(n):
            if is_safe(board, row, col):
                board[row][col] = 1
                if solve(board, col + 1):
                    return True
                board[row][col] = 0

        return False

    def is_safe(board, row, col):
        for i in range(col):
            if board[row][i] == 1:
                return False

        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False

        for i, j in zip(range(row, n, 1), range(col, -1, -1)):
            if board[i][j] == 1:
                return False

        return True

    board = [[0] * n for _ in range(n)]
    if not solve(board, 0):
        return []

    solutions = []
    for row in board:
        solutions.append([1 if cell == 1 else 0 for cell in row])
    return solutions
```

**Power Set**

```python
def power_set(nums):
    def backtrack(subset, i):
        if i == len(nums):
            result.append(subset[:])
            return
        subset.append(nums[i])
        backtrack(subset, i + 1)
        subset.pop()
        backtrack(subset, i + 1)

    result = []
    backtrack([], 0)
    return result
```

**Rat in a Maze**

```python
def solve_maze(maze):
    n = len(maze)

    def solve(maze, x, y, sol):
        if x == n - 1 and y == n - 1 and maze[x][y] == 1:
            sol[x][y] = 1
            return True

        if x >= 0 and y >= 0 and x < n and y < n and maze[x][y] == 1 and sol[x][y] == 0:
            sol[x][y] = 1
            if solve(maze, x + 1, y, sol) or solve(maze, x, y + 1, sol):
                return True
            sol[x][y] = 0
        return False

    sol = [[0] * n for _ in range(n)]
    if not solve(maze, 0, 0, sol):
        print("Solution doesn't exist")
        return False
    return sol
```

**Permutations of a String/Array**

```python
def permutations(nums):
    def backtrack(nums, index, perm):
        if index == len(nums):
            result.append(perm[:])
            return

        for i in range(index, len(nums)):
            nums[index], nums[i] = nums[i], nums[index]
            backtrack(nums, index + 1, perm + [nums[index]])
            nums[index], nums[i] = nums[i], nums[index]

    result = []
    backtrack(nums, 0, [])
    return result
```

**Word Search**

```python
def word_search(board, word):
    def dfs(i, j, word_index):
        if word_index == len(word):
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[word_index]:
            return False
        temp = board[i][j]
        board[i][j] = '#'
        found = dfs(i + 1, j, word_index + 1) or \
                dfs(i - 1, j, word_index + 1) or \
                dfs(i, j + 1, word_index + 1) or \
                dfs(i, j - 1, word_index + 1)
        board[i][j] = temp
        return found

    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(i, j, 0):
                return True
    return False
```

**Sudoku Solver**

```python
def solve_sudoku(board):
    def find_empty_cell(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return i, j
        return -1, -1

    def is_valid(board, row, col, num):
        for i in range(9):
            if board[i][col] == num:
                return False
        for j in range(9):
            if board[row][j] == num:
                return False
        start_row, start_col = row - row % 3, col - col % 3
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False
        return True

    row, col = find_empty_cell(board)
    if row == -1:
        return True

    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False
```

**Partition a String into Palindromic Substrings**

```python
def partition_palindromes(s):
    def is_palindrome(s, start, end):
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    def backtrack(i, path):
        if i == len(s):
            result.append(path[:])
            return
        for j in range(i, len(s)):
            if is_palindrome(s, i, j):
                path.append(s[i:j+1])
                backtrack(j+1, path)
                path.pop()

    result = []
    backtrack(0, [])
    return result
```

**Print All Combinations of Well-Formed Parentheses**

```python
def generate_parentheses(n):
    def backtrack(s, left, right):
        if len(s) == 2 * n:
            result.append(s)
            return
        if left < n:
            backtrack(s + '(', left + 1, right)
        if right < left:
            backtrack(s + ')', left, right + 1)

    result = []
    backtrack("", 0, 0)
    return result
```

**1. Fibonacci Sequence**

**Memoization:**

```python
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

**Tabulation:**

```python
def fib_tabulation(n):
    dp = [0] * (n + 1)
    dp[0], dp[1] = 0, 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

**2. Longest Increasing Subsequence (LIS)**

```python
def lis(nums):
    n = len(nums)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
```

**3. Longest Common Subsequence (LCS)**

```python
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]
```

**4. 0/1 Knapsack Problem**

```python
def knapsack(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][W]
```

**5. Minimum Number of Coins for a Given Amount (Coin Change Problem)**

```python
def min_coins(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

**6. Edit Distance Between Two Strings**

```python
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
```

**7. Partition a Set into Two Subsets with Minimum Difference**

```python
def min_subset_sum_difference(nums):
    total_sum = sum(nums)
    target = total_sum // 2
    dp = [[False] * (target + 1) for _ in range(len(nums) + 1)]
    dp[0][0] = True
    for i in range(1, len(nums) + 1):
        for j in range(target + 1):
            if j == 0:
                dp[i][j] = True
            elif nums[i - 1] <= j:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]
    diff = float('inf')
    for j in range(target + 1):
        if dp[len(nums)][j]:
            diff = min(diff, abs(total_sum - 2 * j))
    return diff
```

**8. Maximum Profit in Stock Trading (Buy/Sell Stock Problems)**

```python
# Single transaction
def max_profit_single_transaction(prices):
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit

# Multiple transactions
def max_profit_multiple_transactions(prices):
    profit = 0
    for i in range(1, len(prices)):
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]
    return profit
```

**9. Subset Sum Problem**

```python
def subset_sum(nums, target):
    n = len(nums)
    dp = [[False] * (target + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = True
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            if nums[i - 1] <= j:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
            else:
                dp[i][j] = dp[i - 1][j]
    return dp[n][target]
```

**10. Count the Number of Ways to Decode a String**

```python
def num_decodings(s):
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1 if s[0] != '0' else 0
    for i in range(2, n + 1):
        if s[i - 1] != '0':
            dp[i] += dp[i - 1]
        if s[i - 2] == '1' or (s[i - 2] == '2' and s[i - 1] <= '6'):
            dp[i] += dp[i - 2]
    return dp[n]
```


**1. Sorting Algorithms**

**Merge Sort**

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    return merge(left_half, right_half)

def merge(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged += left[i:]
    merged += right[j:]
    return merged
```

**Quick Sort**

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)
```

**2. Kth Largest/Smallest Element**

```python
def kth_largest(arr, k):
    return sorted(arr)[-k]

def kth_smallest(arr, k):
    return sorted(arr)[k - 1]
```

**3. Binary Search and Variations**

**Binary Search**

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**First/Last Occurrence**

```python
def first_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            if mid == 0 or arr[mid - 1] != target:
                return mid
            else:
                right = mid - 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def last_occurrence(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            if mid == len(arr) - 1 or arr[mid + 1] != target:
                return mid
            else:
                left = mid + 1
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**Peak Element**

```python
def find_peak_element(arr):
    left, right = 0, len(arr) - 1
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < arr[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left
```

**Square Root of a Number**

```python
def sqrt(x):
    if x == 0 or x == 1:
        return x
    left, right = 1, x
    while left <= right:
        mid = (left + right) // 2
        if mid * mid == x:
            return mid
        elif mid * mid < x:
            left = mid + 1
        else:
            right = mid - 1
    return right
```

**Search in a Matrix**

```python
def search_matrix(matrix, target):
    m, n = len(matrix), len(matrix[0])
    i, j = 0, n - 1
    while i < m and j >= 0:
        if matrix[i][j] == target:
            return True
        elif matrix[i][j] > target:
            j -= 1
        else:
            i += 1
    return False
```

**Median of Two Sorted Arrays**

```python
def find_median_sorted_arrays(nums1, nums2):
    A, B = nums1, nums2
    total = len(A) + len(B)
    half = total // 2

    if len(B) < len(A):
        A, B = B, A

    l, r = 0, len(A) - 1
    while True:
        A_left = l
        B_left = half - A_left

        A_right = A_left + 1
        B_right = total - A_right

        A_left_val = A[A_left] if A_left >= 0 else float('-infinity')
        A_right_val = A[A_right] if A_right < len(A) else float('infinity')
        B_left_val = B[B_left] if B_left >= 0 else float('-infinity')
        B_right_val = B[B_right] if B_right < len(B) else float('infinity')

        if A_left_val <= B_right_val and B_left_val <= A_right_val:
            if total % 2 == 0:
                return (max(A_left_val, B_left_val) + min(A_right_val, B_right_val)) / 2
            else:
                return max(A_left_val, B_left_val)
        elif A_left_val > B_right_val:
            r = A_left - 1
        else:
            l = A_left + 1
```


**1. Find Two Numbers that Add Up to a Given Sum (Hash Map)**

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**2. Group Anagrams from a List of Strings**

```python
def group_anagrams(strs):
    groups = {}
    for s in strs:
        sorted_s = ''.join(sorted(s))
        if sorted_s not in groups:
            groups[sorted_s] = []
        groups[sorted_s].append(s)
    return list(groups.values())
```

**3. Find the Longest Subarray with Sum Equal to k**

```python
def longest_subarray_with_sum_k(nums, k):
    prefix_sum = 0
    max_len = 0
    prefix_sum_to_index = {0: -1}
    for i, num in enumerate(nums):
        prefix_sum += num
        if prefix_sum - k in prefix_sum_to_index:
            max_len = max(max_len, i - prefix_sum_to_index[prefix_sum - k])
        if prefix_sum not in prefix_sum_to_index:
            prefix_sum_to_index[prefix_sum] = i
    return max_len
```

**4. Count Distinct Elements in Every Window of Size k**

```python
from collections import defaultdict

def count_distinct_elements(arr, k):
    n = len(arr)
    freq = defaultdict(int)
    distinct_count = 0
    result = []

    for i in range(k):
        if freq[arr[i]] == 0:
            distinct_count += 1
        freq[arr[i]] += 1

    result.append(distinct_count)

    for i in range(k, n):
        if freq[arr[i - k]] == 1:
            distinct_count -= 1
        freq[arr[i - k]] -= 1
        if freq[arr[i]] == 0:
            distinct_count += 1
        freq[arr[i]] += 1
        result.append(distinct_count)

    return result
```

**5. Check if Two Strings are Anagrams**

```python
def is_anagram(s, t):
    if len(s) != len(t):
        return False
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    for char in t:
        if char not in char_count or char_count[char] == 0:
            return False
        char_count[char] -= 1
    return True
```

**6. Implement an LRU (Least Recently Used) Cache**

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```


**1. Implement a Trie (Prefix Tree)**

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

**2. Union-Find (Disjoint Set Union) with Path Compression**

```python
class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
```

**3. Heap (Min-Heap)**

```python
import heapq

class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, value):
        heapq.heappush(self.heap, value)

    def pop(self):
        return heapq.heappop(self.heap)

    def top(self):
        return self.heap[0]

    def is_empty(self):
        return len(self.heap) == 0
```

**4. Sliding Window Technique (Maximum Sum Subarray of Size k)**

```python
def max_sum_subarray(nums, k):
    n = len(nums)
    if n < k:
        return -1

    max_sum = current_sum = sum(nums[:k])
    for i in range(k, n):
        current_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, current_sum)
    return max_sum
```

**5. Fenwick Tree (Binary Indexed Tree)**

```python
class FenwickTree:
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, index, value):
        index += 1
        while index <= self.n:
            self.tree[index] += value
            index += index & (-index)

    def query(self, index):
        index += 1
        sum = 0
        while index > 0:
            sum += self.tree[index]
            index -= index & (-index)
        return sum
```

**6. Count Inversions in an Array (Merge Sort-Based Approach)**

```python
def merge_sort_and_count(arr):
    if len(arr) <= 1:
        return arr, 0

    mid = len(arr) // 2
    left_half, inv_count_left = merge_sort_and_count(arr[:mid])
    right_half, inv_count_right = merge_sort_and_count(arr[mid:])

    merged, inv_count_merge = merge_and_count_inversions(left_half, right_half)
    return merged, inv_count_left + inv_count_right + inv_count_merge

def merge_and_count_inversions(left, right):
    merged = []
    i = j = inv_count = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inv_count += len(left) - i
            j += 1
    merged += left[i:]
    merged += right[j:]
    return merged, inv_count
```


## Tips for Solving DSA Problems

**1. Understand the Problem Statement:**

* **Read carefully:** Make sure you fully comprehend the problem, including inputs, outputs, and constraints.
* **Break it down:** Divide the problem into smaller, more manageable subproblems.
* **Identify the core data structures and algorithms:** Determine which data structures and algorithms are best suited for the problem.

**2. Choose the Right Data Structure:**

* **Array:** Use for storing and accessing elements sequentially.
* **Stack:** Use for Last-In-First-Out (LIFO) operations.
* **Queue:** Use for First-In-First-Out (FIFO) operations.
* **Linked List:** Use for dynamic data structures where elements can be inserted or deleted efficiently.
* **Tree:** Use for hierarchical data structures.
* **Hash Map:** Use for efficient key-value lookup.
* **Heap:** Use for priority queue operations.

**3. Optimize Time and Space Complexity:**

* **Brute Force:** Start with a simple, straightforward solution.
* **Optimize Algorithms:** Use efficient algorithms like dynamic programming, greedy algorithms, divide and conquer, and backtracking.
* **Analyze Time and Space Complexity:** Use Big O notation to measure the efficiency of your solution.
* **Identify Bottlenecks:** Find the parts of your code that are slowing it down and optimize them.

**4. Write Clean Code:**

* **Use meaningful variable names:** Make your code readable.
* **Add comments:** Explain the purpose of code sections.
* **Modularize your code:** Break down complex problems into smaller functions.
* **Handle edge cases:** Consider all possible input scenarios.
* **Test your code:** Write unit tests to ensure correctness.

**5. Practice Regularly:**

* **Use online platforms:** LeetCode, Codeforces, GeeksforGeeks, and HackerRank are excellent resources for practicing DSA problems.
* **Solve a variety of problems:** Practice different problem types to improve your problem-solving skills.
* **Analyze solutions:** Learn from others' solutions and understand different approaches.
* **Participate in coding challenges:** Compete with others to improve your speed and accuracy.

By following these tips, you can enhance your problem-solving skills and become proficient in data structures and algorithms.
