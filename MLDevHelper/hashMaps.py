import pandas as pd


class TreeNode:
    def __init__(self, obj):
        self.obj = obj
        self.left = None
        self.right = None
        self.height = 1


class AVL_Tree:
    def __init__(self):
        self.root = None

    def insert(self, obj):
        def insert_helper(root, obj):
            if not root:
                return TreeNode(obj)
            elif obj < root.obj:
                root.left = insert_helper(root.left, obj)
            else:
                root.right = insert_helper(root.right, obj)

            root.height = 1 + max(
                self._get_height(root.left), self._get_height(root.right)
            )
            balance = self._get_balance(root)

            if balance > 1 and obj < root.left.obj:
                return self._right_rotate(root)

            if balance < -1 and obj > root.right.obj:
                return self._left_rotate(root)

            if balance > 1 and obj > root.left.obj:
                root.left = self._left_rotate(root.left)
                return self._right_rotate(root)

            if balance < -1 and obj < root.right.obj:
                root.right = self._right_rotate(root.right)
                return self._left_rotate(root)

            return root

        self.root = insert_helper(self.root, obj)

    def delete(self, obj):
        def delete_helper(root, obj):
            if not root:
                return root

            if obj < root.obj:
                root.left = delete_helper(root.left, obj)
            elif obj > root.obj:
                root.right = delete_helper(root.right, obj)
            else:
                if not root.left or not root.right:
                    temp = root.left if root.left else root.right
                    root = temp
                else:
                    temp = self._get_min_value_node(root.right)
                    root.obj = temp.obj
                    root.right = delete_helper(root.right, temp.obj)

            if not root:
                return root

            root.height = 1 + max(
                self._get_height(root.left), self._get_height(root.right)
            )
            balance = self._get_balance(root)

            if balance > 1 and self._get_balance(root.left) >= 0:
                return self._right_rotate(root)

            if balance > 1 and self._get_balance(root.left) < 0:
                root.left = self._left_rotate(root.left)
                return self._right_rotate(root)

            if balance < -1 and self._get_balance(root.right) <= 0:
                return self._left_rotate(root)

            if balance < -1 and self._get_balance(root.right) > 0:
                root.right = self._right_rotate(root.right)
                return self._left_rotate(root)

            return root

        self.root = delete_helper(self.root, obj)

    def search(self, obj):
        def search_helper(root, obj):
            if not root:
                return None  # If root is None, return None

            if obj == root.obj:
                return root.obj  # If obj matches, return the object in the node

            if obj < root.obj:
                return search_helper(root.left, obj)
            else:
                return search_helper(root.right, obj)

        result = search_helper(self.root, obj)
        return result

    def _left_rotate(self, z):
        y = z.right
        T2 = y.left

        y.left = z
        z.right = T2

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _right_rotate(self, z):
        y = z.left
        T3 = y.right

        y.right = z
        z.left = T3

        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        return y

    def _get_height(self, root):
        if not root:
            return 0
        return root.height

    def _get_balance(self, root):
        if not root:
            return 0
        return self._get_height(root.left) - self._get_height(root.right)

    def _get_min_value_node(self, root):
        current = root
        while current.left:
            current = current.left
        return current

    def inorder(self):
        a = []

        def inorder_helper(root, a):
            if root:
                inorder_helper(root.left, a)
                a.append((root.obj.image_path, root.obj.label))
                inorder_helper(root.right, a)

        inorder_helper(self.root, a)
        return a


class TrainingExample:
    def __init__(self, image_path, label):
        self.image_path = image_path
        self.label = label

    def __lt__(self, other):
        return self.image_path < other.image_path

    def __gt__(self, other):
        return self.image_path > other.image_path

    def __eq__(self, other):
        return self.image_path == other.image_path


class HashMaps:
    def __init__(self, capacity=300):
        self.__capacity = capacity
        self.__buckets = [None] * capacity

    def __hash(self, image_path):
        return hash(image_path) % self.__capacity

    def insert(self, image_path, label):
        obj = TrainingExample(image_path, label)
        index = self.__hash(image_path)
        if not self.__buckets[index]:
            self.__buckets[index] = AVL_Tree()
        self.__buckets[index].insert(obj)

    def get(self, image_path):
        index = self.__hash(image_path)
        if self.__buckets[index]:
            return self.__buckets[index].search(TrainingExample(image_path, ""))
        return None

    def remove(self, image_path):
        index = self.__hash(image_path)
        if self.__buckets[index]:
            self.__buckets[index].delete(TrainingExample(image_path, ""))

    def bulk_insert_from_dataframe(self, df):
        for index, row in df.iterrows():
            self.insert(row["images"], row["labels"])

    def to_dataframe(self):
        data = {"images": [], "labels": []}
        for tree in self.__buckets:
            if tree:
                data = self._collect_data_from_tree(tree.root, data)
        return pd.DataFrame(data)

    def _collect_data_from_tree(self, root, data):
        if root:
            data = self._collect_data_from_tree(root.left, data)
            data["images"].append(root.obj.image_path)
            data["labels"].append(root.obj.label)
            data = self._collect_data_from_tree(root.right, data)
        return data

    def display_map(self):
        for each_buc in self.__buckets:
            if each_buc:
                print(each_buc.inorder())
