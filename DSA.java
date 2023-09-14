// Declaration and initialization
int[] numbers = new int[5];  // Creates an integer array of size 5

// Initialization after declaration
numbers[0] = 10;
numbers[1] = 20;
numbers[2] = 30;
numbers[3] = 40;
numbers[4] = 50;

// Accessing Array Elements:
int secondNumber = numbers[1];  // Accesses the second element (20)

// Modifying Array Elements: 
numbers[2] = 35;  // Changes the third element's value to 35

// Array Length
int length = numbers.length;  // Gets the length of the array (5 in this case)

// Iterating Through an Array:
for (int i = 0; i < numbers.length; i++) {
    System.out.println(numbers[i]);  // Prints each element
}

// Multidimensional Arrays: 
int[][] matrix = new int[3][4];  // Creates a 3x4 matrix

matrix[1][2] = 42;  // Accesses and modifies an element in the matrix

// Array Copying:
int[] copy = new int[numbers.length];
System.arraycopy(numbers, 0, copy, 0, numbers.length);

// Sorting an Array:
Arrays.sort(numbers);  // Sorts the array in ascending order
sort(numbers);  // Sorts the array in ascending order

// Linked List Implementation: 
class Node {
    int data;
    Node next;

    public Node(int data) {
        this.data = data;
        this.next = null;
    }
}

class SinglyLinkedList {
    Node head;

    public void insert(int data) {
        Node newNode = new Node(data);
        if (head == null) {
            head = newNode;
        } else {
            Node current = head;
            while (current.next != null) {
                current = current.next;
            }
            current.next = newNode;
        }
    }

    public void display() {
        Node current = head;
        while (current != null) {
            System.out.print(current.data + " ");
            current = current.next;
        }
        System.out.println();
    }
}

public class Main {
    public static void main(String[] args) {
        SinglyLinkedList list = new SinglyLinkedList();
        list.insert(10);
        list.insert(20);
        list.insert(30);
        list.display(); // Output: 10 20 30
    }
}

// LIFO principle with a simple example using a stack:
import java.util.Stack;

public class LIFOPrincipleExample {
    public static void main(String[] args) {
        Stack<Integer> stack = new Stack<>();

        // Push elements onto the stack
        stack.push(10);
        stack.push(20);
        stack.push(30);

        // Pop elements from the stack
        int popped1 = stack.pop(); // Removes 30
        int popped2 = stack.pop(); // Removes 20

        System.out.println("Popped elements: " + popped1 + ", " + popped2);
    }
}

// Implementing a Stac
class Stack {
    private int maxSize;
    private int[] stackArray;
    private int top;

    public Stack(int size) {
        maxSize = size;
        stackArray = new int[maxSize];
        top = -1; // Stack is initially empty
    }

    public void push(int value) {
        if (top < maxSize - 1) {
            stackArray[++top] = value;
            System.out.println("Pushed: " + value);
        } else {
            System.out.println("Stack is full. Cannot push " + value);
        }
    }

    public int pop() {
        if (top >= 0) {
            int value = stackArray[top--];
            System.out.println("Popped: " + value);
            return value;
        } else {
            System.out.println("Stack is empty. Cannot pop.");
            return -1; // Return a sentinel value indicating failure
        }
    }

    public boolean isEmpty() {
        return top == -1;
    }

    public int size() {
        return top + 1;
    }
}

public class StackExample {
    public static void main(String[] args) {
        Stack stack = new Stack(5);

        stack.push(10);
        stack.push(20);
        stack.push(30);

        System.out.println("Stack size: " + stack.size());

        int poppedValue = stack.pop();
        System.out.println("Popped value: " + poppedValue);

        System.out.println("Is stack empty? " + stack.isEmpty());

        stack.push(40);
        stack.push(50);

        System.out.println("Stack size: " + stack.size());

        while (!stack.isEmpty()) {
            stack.pop();
        }

        System.out.println("Is stack empty? " + stack.isEmpty());
    }
}

// implementing a queue.
import java.util.LinkedList;

class Queue {
    private LinkedList<Integer> list;

    public Queue() {
        list = new LinkedList<>();
    }

    public void enqueue(int value) {
        list.addLast(value);
        System.out.println("Enqueued: " + value);
    }

    public int dequeue() {
        if (!isEmpty()) {
            int value = list.removeFirst();
            System.out.println("Dequeued: " + value);
            return value;
        } else {
            System.out.println("Queue is empty. Cannot dequeue.");
            return -1; // Return a sentinel value indicating failure
        }
    }

    public boolean isEmpty() {
        return list.isEmpty();
    }

    public int size() {
        return list.size();
    }
}

public class QueueExample {
    public static void main(String[] args) {
        Queue queue = new Queue();

        queue.enqueue(10);
        queue.enqueue(20);
        queue.enqueue(30);

        System.out.println("Queue size: " + queue.size());

        int dequeuedValue = queue.dequeue();
        System.out.println("Dequeued value: " + dequeuedValue);

        System.out.println("Is queue empty? " + queue.isEmpty());

        queue.enqueue(40);
        queue.enqueue(50);

        System.out.println("Queue size: " + queue.size());

        while (!queue.isEmpty()) {
            queue.dequeue();
        }

        System.out.println("Is queue empty? " + queue.isEmpty());
    }
}

// implementing a binary tree.
class Node {
    int data;
    Node left;
    Node right;

    public Node(int data) {
        this.data = data;
        this.left = null;
        this.right = null;
    }
}

public class BinaryTreeExample {
    public static void main(String[] args) {
        Node root = new Node(10);
        root.left = new Node(5);
        root.right = new Node(15);
        root.left.left = new Node(3);
        root.left.right = new Node(7);
        root.right.right = new Node(20);

        // Perform tree traversal
        System.out.println("Inorder Traversal:");
        inorderTraversal(root);

        System.out.println("\nPreorder Traversal:");
        preorderTraversal(root);

        System.out.println("\nPostorder Traversal:");
        postorderTraversal(root);
    }

    public static void inorderTraversal(Node root) {
        if (root != null) {
            inorderTraversal(root.left);
            System.out.print(root.data + " ");
            inorderTraversal(root.right);
        }
    }

    public static void preorderTraversal(Node root) {
        if (root != null) {
            System.out.print(root.data + " ");
            preorderTraversal(root.left);
            preorderTraversal(root.right);
        }
    }

    public static void postorderTraversal(Node root) {
        if (root != null) {
            postorderTraversal(root.left);
            postorderTraversal(root.right);
            System.out.print(root.data + " ");
        }
    }
}

// implementing a Binary Search Tree
class Node {
    int key;
    Node left;
    Node right;

    public Node(int key) {
        this.key = key;
        this.left = null;
        this.right = null;
    }
}

public class BinarySearchTreeExample {
    static Node insert(Node root, int key) {
        if (root == null) {
            return new Node(key);
        }
        if (key < root.key) {
            root.left = insert(root.left, key);
        } else if (key > root.key) {
            root.right = insert(root.right, key);
        }
        return root;
    }

    static void inorderTraversal(Node root) {
        if (root != null) {
            inorderTraversal(root.left);
            System.out.print(root.key + " ");
            inorderTraversal(root.right);
        }
    }

    public static void main(String[] args) {
        Node root = null;
        int[] keys = {15, 10, 20, 8, 12, 17, 25};

        for (int key : keys) {
            root = insert(root, key);
        }

        System.out.println("Inorder Traversal:");
        inorderTraversal(root);
    }
}


//Avl tree
class Node {
    int key;
    int height;
    Node left;
    Node right;

    public Node(int key) {
        this.key = key;
        this.height = 1;
        this.left = null;
        this.right = null;
    }
}

public class AVLTreeExample {
    static int height(Node node) {
        if (node == null) {
            return 0;
        }
        return node.height;
    }

    static int balanceFactor(Node node) {
        if (node == null) {
            return 0;
        }
        return height(node.left) - height(node.right);
    }

    static Node rightRotate(Node y) {
        Node x = y.left;
        Node T2 = x.right;

        x.right = y;
        y.left = T2;

        y.height = Math.max(height(y.left), height(y.right)) + 1;
        x.height = Math.max(height(x.left), height(x.right)) + 1;

        return x;
    }

    // Perform left rotation
    // ... (implementation similar to rightRotate)

    // Perform double rotation (left-right rotation)
    // ... (implementation using rotations)

    // Perform double rotation (right-left rotation)
    // ... (implementation using rotations)

    static Node insert(Node root, int key) {
        if (root == null) {
            return new Node(key);
        }
        if (key < root.key) {
            root.left = insert(root.left, key);
        } else if (key > root.key) {
            root.right = insert(root.right, key);
        } else {
            return root; // Duplicate keys not allowed
        }

        // Update height of current node
        root.height = 1 + Math.max(height(root.left), height(root.right));

        int balance = balanceFactor(root);

        // Perform rotations if needed to restore balance
        // ... (implementation of rotations)

        return root;
    }

    // ... (other operations like search, traversal, etc.)

    public static void main(String[] args) {
        Node root = null;
        int[] keys = {10, 5, 15, 3, 7, 12, 20};

        for (int key : keys) {
            root = insert(root, key);
        }

        // Perform AVL tree operations as needed
    }
}

//graph dfs recurssion
import java.util.*;

class Graph {
    private int V; // Number of vertices
    private LinkedList<Integer>[] adjList;

    public Graph(int vertices) {
        V = vertices;
        adjList = new LinkedList[V];
        for (int i = 0; i < V; i++) {
            adjList[i] = new LinkedList<>();
        }
    }

    public void addEdge(int v, int w) {
        adjList[v].add(w);
    }

    public void DFS(int vertex, boolean[] visited) {
        visited[vertex] = true;
        System.out.print(vertex + " ");

        for (Integer neighbor : adjList[vertex]) {
            if (!visited[neighbor]) {
                DFS(neighbor, visited);
            }
        }
    }

    public void performDFS(int startVertex) {
        boolean[] visited = new boolean[V];
        DFS(startVertex, visited);
    }
}

//dfs traversal
public class DFSExample {
    public static void main(String[] args) {
        Graph graph = new Graph(7);

        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);

        System.out.println("Depth-First Traversal (starting from vertex 0):");
        graph.performDFS(0);
    }
}

//graph bfs 
import java.util.*;

class Graph {
    private int V; // Number of vertices
    private LinkedList<Integer>[] adjList;

    public Graph(int vertices) {
        V = vertices;
        adjList = new LinkedList[V];
        for (int i = 0; i < V; i++) {
            adjList[i] = new LinkedList<>();
        }
    }

    public void addEdge(int v, int w) {
        adjList[v].add(w);
    }

    public void BFS(int startVertex) {
        boolean[] visited = new boolean[V];
        Queue<Integer> queue = new LinkedList<>();

        visited[startVertex] = true;
        queue.add(startVertex);

        while (!queue.isEmpty()) {
            int vertex = queue.poll();
            System.out.print(vertex + " ");

            for (Integer neighbor : adjList[vertex]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.add(neighbor);
                }
            }
        }
    }
}

//bfsn traversal
public class BFSExample {
    public static void main(String[] args) {
        Graph graph = new Graph(7);

        graph.addEdge(0, 1);
        graph.addEdge(0, 2);
        graph.addEdge(1, 3);
        graph.addEdge(1, 4);
        graph.addEdge(2, 5);
        graph.addEdge(2, 6);

        System.out.println("Breadth-First Traversal (starting from vertex 0):");
        graph.BFS(0);
    }
}


//bubble sort 
public class BubbleSortExample {
    static void bubbleSort(int[] array) {
        int n = array.length;
        boolean swapped;

        for (int i = 0; i < n - 1; i++) {
            swapped = false;
            for (int j = 0; j < n - i - 1; j++) {
                if (array[j] > array[j + 1]) {
                    // Swap array[j] and array[j+1]
                    int temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                    swapped = true;
                }
            }

            // If no two elements were swapped in the inner loop, the array is sorted
            if (!swapped) {
                break;
            }
        }
    }

    public static void main(String[] args) {
        int[] arr = {64, 34, 25, 12, 22, 11, 90};
        System.out.println("Original array: " + Arrays.toString(arr));

        bubbleSort(arr);

        System.out.println("Sorted array: " + Arrays.toString(arr));
    }
}

//selecction sort

import java.util.Arrays;

public class SelectionSortExample {
    static void selectionSort(int[] array) {
        int n = array.length;

        for (int i = 0; i < n - 1; i++) {
            int minIndex = i;

            // Find the index of the minimum element in the unsorted part
            for (int j = i + 1; j < n; j++) {
                if (array[j] < array[minIndex]) {
                    minIndex = j;
                }
            }

            // Swap the found minimum element with the first element of the unsorted part
            int temp = array[minIndex];
            array[minIndex] = array[i];
            array[i] = temp;
        }
    }

    public static void main(String[] args) {
        int[] arr = {64, 25, 12, 22, 11};
        System.out.println("Original array: " + Arrays.toString(arr));

        selectionSort(arr);

        System.out.println("Sorted array: " + Arrays.toString(arr));
    }
}

//insertion sort
import java.util.Arrays;

public class InsertionSortExample {
    static void insertionSort(int[] array) {
        int n = array.length;

        for (int i = 1; i < n; i++) {
            int key = array[i];
            int j = i - 1;

            // Move elements of the sorted portion that are greater than key to the right
            while (j >= 0 && array[j] > key) {
                array[j + 1] = array[j];
                j--;
            }

            // Insert the key into its correct position
            array[j + 1] = key;
        }
    }

    public static void main(String[] args) {
        int[] arr = {64, 25, 12, 22, 11};
        System.out.println("Original array: " + Arrays.toString(arr));

        insertionSort(arr);

        System.out.println("Sorted array: " + Arrays.toString(arr));
    }
}

//merge sort
import java.util.Arrays;

public class MergeSortExample {
    static void merge(int[] array, int left, int middle, int right) {
        int n1 = middle - left + 1;
        int n2 = right - middle;

        int[] leftArray = new int[n1];
        int[] rightArray = new int[n2];

        for (int i = 0; i < n1; i++) {
            leftArray[i] = array[left + i];
        }
        for (int j = 0; j < n2; j++) {
            rightArray[j] = array[middle + 1 + j];
        }

        int i = 0, j = 0, k = left;
        while (i < n1 && j < n2) {
            if (leftArray[i] <= rightArray[j]) {
                array[k] = leftArray[i];
                i++;
            } else {
                array[k] = rightArray[j];
                j++;
            }
            k++;
        }

        while (i < n1) {
            array[k] = leftArray[i];
            i++;
            k++;
        }

        while (j < n2) {
            array[k] = rightArray[j];
            j++;
            k++;
        }
    }

    static void mergeSort(int[] array, int left, int right) {
        if (left < right) {
            int middle = left + (right - left) / 2;

            mergeSort(array, left, middle);
            mergeSort(array, middle + 1, right);

            merge(array, left, middle, right);
        }
    }

    public static void main(String[] args) {
        int[] arr = {64, 25, 12, 22, 11};
        System.out.println("Original array: " + Arrays.toString(arr));

        mergeSort(arr, 0, arr.length - 1);

        System.out.println("Sorted array: " + Arrays.toString(arr));
    }
}

//quick sort
import java.util.Arrays;

public class QuickSortExample {
    static int partition(int[] array, int low, int high) {
        int pivot = array[high];
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (array[j] < pivot) {
                i++;

                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        int temp = array[i + 1];
        array[i + 1] = array[high];
        array[high] = temp;

        return i + 1;
    }

    static void quickSort(int[] array, int low, int high) {
        if (low < high) {
            int pivotIndex = partition(array, low, high);

            quickSort(array, low, pivotIndex - 1);
            quickSort(array, pivotIndex + 1, high);
        }
    }

    public static void main(String[] args) {
        int[] arr = {64, 25, 12, 22, 11};
        System.out.println("Original array: " + Arrays.toString(arr));

        quickSort(arr, 0, arr.length - 1);

        System.out.println("Sorted array: " + Arrays.toString(arr));
    }
}

//search
int[] arr = {2, 5, 8, 12, 16, 23, 38, 56, 72, 91};

//linear search
int target = 23;
int index = -1;

for (int i = 0; i < arr.length; i++) {
    if (arr[i] == target) {
        index = i;
        break;
    }
}

if (index != -1) {
    System.out.println("Linear Search: Element found at index " + index);
} else {
    System.out.println("Linear Search: Element not found");
}

//binary search
int target = 23;
int low = 0;
int high = arr.length - 1;
int index = -1;

while (low <= high) {
    int mid = low + (high - low) / 2;
    
    if (arr[mid] == target) {
        index = mid;
        break;
    } else if (arr[mid] < target) {
        low = mid + 1;
    } else {
        high = mid - 1;
    }
}

if (index != -1) {
    System.out.println("Binary Search: Element found at index " + index);
} else {
    System.out.println("Binary Search: Element not found");
}

//hashing
class HashTable {
    private String[] table;
    private int size;

    public HashTable(int size) {
        this.size = size;
        this.table = new String[size];
    }

    public int hash(String key) {
        int sum = 0;
        for (char c : key.toCharArray()) {
            sum += c;
        }
        return sum % size;
    }

    public void insert(String key, String value) {
        int index = hash(key);
        table[index] = value;
    }

    public String get(String key) {
        int index = hash(key);
        return table[index];
    }
}

public class HashingExample {
    public static void main(String[] args) {
        HashTable hashTable = new HashTable(10);

        hashTable.insert("John", "555-1234");
        hashTable.insert("Jane", "555-5678");
        hashTable.insert("Alice", "555-9876");

        System.out.println("John's phone number: " + hashTable.get("John"));
        System.out.println("Alice's phone number: " + hashTable.get("Alice"));
        System.out.println("Jane's phone number: " + hashTable.get("Jane"));
    }
}


//hash table
class HashTable {
    private int size;
    private String[] data;

    public HashTable(int size) {
        this.size = size;
        this.data = new String[size];
    }

    private int hash(String key) {
        int hash = 0;
        for (char c : key.toCharArray()) {
            hash += c;
        }
        return hash % size;
    }

    public void put(String key, String value) {
        int index = hash(key);
        data[index] = value;
    }

    public String get(String key) {
        int index = hash(key);
        return data[index];
    }
}

public class HashTableExample {
    public static void main(String[] args) {
        HashTable hashTable = new HashTable(10);

        hashTable.put("John", "555-1234");
        hashTable.put("Jane", "555-5678");

        System.out.println("John's phone number: " + hashTable.get("John"));
        System.out.println("Jane's phone number: " + hashTable.get("Jane"));
    }
}

//chaining 
class HashTable {
    private int size;
    private LinkedList<Pair>[] buckets;

    public HashTable(int size) {
        this.size = size;
        this.buckets = new LinkedList[size];
        for (int i = 0; i < size; i++) {
            buckets[i] = new LinkedList<>();
        }
    }

    private int hash(String key) {
        // hash function implementation
    }

    public void put(String key, String value) {
        int index = hash(key);
        buckets[index].add(new Pair(key, value));
    }

    public String get(String key) {
        int index = hash(key);
        for (Pair pair : buckets[index]) {
            if (pair.key.equals(key)) {
                return pair.value;
            }
        }
        return null;
    }

    private class Pair {
        String key;
        String value;

        Pair(String key, String value) {
            this.key = key;
            this.value = value;
        }
    }
}

//open addressing
class HashTable {
    private int size;
    private String[] data;

    public HashTable(int size) {
        this.size = size;
        this.data = new String[size];
    }

    private int hash(String key) {
        // hash function implementation
    }

    public void put(String key, String value) {
        int index = hash(key);

        // Linear probing
        while (data[index] != null) {
            index = (index + 1) % size;
        }

        data[index] = value;
    }

    public String get(String key) {
        int index = hash(key);

        // Linear probing
        while (data[index] != null && !data[index].equals(key)) {
            index = (index + 1) % size;
        }

        return data[index];
    }
}

//recursion
public class Main {
  public static void main(String[] args) {
    int result = sum(10);
    System.out.println(result);
  }
  public static int sum(int k) {
    if (k > 0) {
      return k + sum(k - 1);
    } else {
      return 0;
    }
  }
}

//RecursiveFactorial
public class RecursiveFactorial {
    public static int factorial(int n) {
        // Base case: when n is 0 or 1, the factorial is 1
        if (n == 0 || n == 1) {
            return 1;
        } else {
            // Recursive case: n! = n * (n-1)!
            return n * factorial(n - 1);
        }
    }

    public static void main(String[] args) {
        int number = 5;
        int result = factorial(number);
        System.out.println("Factorial of " + number + " is " + result);
    }
}

//computing the nth Fibonacci number using memoization
import java.util.HashMap;
import java.util.Map;

public class Fibonacci {
    private static Map<Integer, Integer> memo = new HashMap<>();

    public static int fibonacci(int n) {
        if (n <= 1) {
            return n;
        }

        if (memo.containsKey(n)) {
            return memo.get(n);
        }

        int result = fibonacci(n - 1) + fibonacci(n - 2);
        memo.put(n, result);
        return result;
    }

    public static void main(String[] args) {
        int n = 10;
        int result = fibonacci(n);
        System.out.println("Fibonacci(" + n + ") = " + result);
    }
}

//Activity Selection Problem using the greedy algorithm
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

class Activity {
    int startTime;
    int endTime;

    Activity(int startTime, int endTime) {
        this.startTime = startTime;
        this.endTime = endTime;
    }
}

public class ActivitySelection {
    public static List<Activity> selectActivities(List<Activity> activities) {
        List<Activity> selectedActivities = new ArrayList<>();

        // Sort activities based on finish times
        Collections.sort(activities, Comparator.comparingInt(a -> a.endTime));

        // Select the first activity
        selectedActivities.add(activities.get(0));

        // Iterate through the remaining activities
        for (int i = 1; i < activities.size(); i++) {
            Activity currentActivity = activities.get(i);
            Activity lastSelectedActivity = selectedActivities.get(selectedActivities.size() - 1);

            // If the current activity can be scheduled without overlapping, select it
            if (currentActivity.startTime >= lastSelectedActivity.endTime) {
                selectedActivities.add(currentActivity);
            }
        }

        return selectedActivities;
    }

    public static void main(String[] args) {
        List<Activity> activities = new ArrayList<>();
        activities.add(new Activity(1, 4));
        activities.add(new Activity(3, 5));
        activities.add(new Activity(0, 6));
        activities.add(new Activity(5, 7));
        activities.add(new Activity(3, 8));
        activities.add(new Activity(5, 9));
        activities.add(new Activity(6, 10));
        activities.add(new Activity(8, 11));

        List<Activity> selectedActivities = selectActivities(activities);

        System.out.println("Selected Activities:");
        for (Activity activity : selectedActivities) {
            System.out.println("[" + activity.startTime + ", " + activity.endTime + "]");
        }
    }
}

//Knapsack Problem
public class Knapsack {
    public static int knapsack(int[] values, int[] weights, int capacity) {
        int n = values.length;
        int[][] dp = new int[n + 1][capacity + 1];

        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= capacity; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 0;
                } else if (weights[i - 1] <= j) {
                    dp[i][j] = Math.max(dp[i - 1][j], values[i - 1] + dp[i - 1][j - weights[i - 1]]);
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        return dp[n][capacity];
    }

    public static void main(String[] args) {
        int[] values = {60, 100, 120};
        int[] weights = {10, 20, 30};
        int capacity = 50;

        int maxValue = knapsack(values, weights, capacity);
        System.out.println("Maximum value: " + maxValue);
    }
}

//Fractional Knapsack Problem
import java.util.Arrays;
import java.util.Comparator;

class Item {
    int value;
    int weight;
    double valueToWeightRatio;

    public Item(int value, int weight) {
        this.value = value;
        this.weight = weight;
        this.valueToWeightRatio = (double) value / weight;
    }
}

public class FractionalKnapsack {
    public static double fractionalKnapsack(int capacity, Item[] items) {
        // Sort items by value-to-weight ratio in descending order
        Arrays.sort(items, Comparator.comparingDouble((Item item) -> item.valueToWeightRatio).reversed());

        double totalValue = 0.0;
        int remainingCapacity = capacity;

        for (Item item : items) {
            if (remainingCapacity >= item.weight) {
                // Entire item can be added to the knapsack
                totalValue += item.value;
                remainingCapacity -= item.weight;
            } else {
                // Fraction of the item can be added
                totalValue += item.valueToWeightRatio * remainingCapacity;
                break; // No more items can be added
            }
        }

        return totalValue;
    }

    public static void main(String[] args) {
        Item[] items = {
            new Item(60, 10),
            new Item(100, 20),
            new Item(120, 30)
        };

        int capacity = 50;

        double maxValue = fractionalKnapsack(capacity, items);
        System.out.println("Maximum achievable value: " + maxValue);
    }
}

DISM /Online /Cleanup-Image /RestoreHealth