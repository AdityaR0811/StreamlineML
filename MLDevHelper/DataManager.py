import os
import pandas as pd
import glob
import pickle
from .hashMaps import HashMaps


class Folder:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.sub_folders = []


class Raw_Data_Folder:
    def __init__(self, name, data, word_number_labels, train_size, test_size, val_size):
        self.name = name
        self.data = data
        self.word_number_labels = word_number_labels
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size


class TraingExample:
    def __init__(self, id, image_path, label):
        self.id = id
        self.image_path = image_path
        self.label = label


class DataManager:
    def __init__(self):
        self.root = Folder("root_folder")
        self._load_data()

    def add_dataset(
        self,
        dataset_name,
        dataset_path,
        train_size,
        test_size,
        val_size,
        accepted_file_types,
        balance=True,
    ):
        dataset_exists = any(
            folder.folder_name == dataset_name for folder in self.root.sub_folders
        )
        if dataset_exists:
            print(
                f"Dataset '{dataset_name}' already exists. Cannot add duplicate datasets."
            )
            return

        new_dataset_folder = Folder(folder_name=dataset_name)
        self.root.sub_folders.append(new_dataset_folder)

        # raw_data_folder = Folder(folder_name="raw_data_folder_" + dataset_name)
        pre_pros_data_folder = Folder(
            folder_name="pre_pros_data_folder_" + dataset_name
        )

        images = []
        labels = []
        classes = enumerate(os.listdir(dataset_path))

        for l, class_name in classes:
            for ext in accepted_file_types:
                images += glob.glob(
                    rf"{os.path.join(dataset_path, class_name)}/**.{ext}",
                    recursive=True,
                )
                labels += [l] * len(
                    glob.glob(
                        rf"{os.path.join(dataset_path, class_name)}/**.{ext}",
                        recursive=True,
                    )
                )

        df = pd.DataFrame({"images": images, "labels": labels})
        df = df.sample(frac=1).reset_index(drop=True)

        raw_data = HashMaps()
        raw_data.bulk_insert_from_dataframe(df)

        word_number_labels = {
            name: label for label, name in enumerate(os.listdir(dataset_path))
        }

        raw_data_folder = Raw_Data_Folder(
            name="raw_data_folder_" + dataset_name,
            data=raw_data,
            word_number_labels=word_number_labels,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
        )

        new_dataset_folder.sub_folders = [raw_data_folder, pre_pros_data_folder]

        total_size = len(df)
        train_size = int(total_size * train_size)
        test_size = int(total_size * test_size)
        val_size = int(total_size * val_size)

        train_df = df[:train_size]
        test_df = df[train_size : train_size + test_size]
        val_df = df[train_size + test_size :]

        if balance:
            train_df = self._balance_classes(train_df)
            val_df = self._balance_classes(val_df)
            test_df = self._balance_classes(test_df)

        train_folder = Folder(folder_name="train_folder_" + dataset_name)
        val_folder = Folder(folder_name="val_folder_" + dataset_name)
        test_folder = Folder(folder_name="test_folder_" + dataset_name)

        train_folder.sub_folders = train_df
        test_folder.sub_folders = test_df
        val_folder.sub_folders = val_df

        pre_pros_data_folder.sub_folders.extend([train_folder, val_folder, test_folder])

        self._save_data()
        print(f"Dataset '{dataset_name}' added successfully.")

    def delete_dataset(self, dataset_name):
        confirm = input(
            f"Are you sure you want to delete dataset '{dataset_name}'? (y/n): "
        )
        if confirm.lower() == "y":
            for folder in self.root.sub_folders:
                if folder.folder_name == dataset_name:
                    self.root.sub_folders.remove(folder)
                    print(f"Dataset '{dataset_name}' has been deleted.")
                    self._save_data()
                    return
            print(f"Dataset '{dataset_name}' not found.")
        else:
            print(f"Deletion of dataset '{dataset_name}' cancelled.")

    def get_train_df(self, dataset_name):
        dataset = self._find_dataset(dataset_name)
        if dataset:
            pre_pros_data_folder = dataset.sub_folders[1]
            train_folder = next(
                (
                    folder
                    for folder in pre_pros_data_folder.sub_folders
                    if "train_folder" in folder.folder_name
                ),
                None,
            )
            if train_folder:
                return train_folder.sub_folders
            else:
                print(f"Train folder for dataset '{dataset_name}' not found.")
        else:
            print(f"Dataset '{dataset_name}' not found.")
        return None

    def get_test_df(self, dataset_name):
        dataset = self._find_dataset(dataset_name)
        if dataset:
            pre_pros_data_folder = dataset.sub_folders[1]
            test_folder = next(
                (
                    folder
                    for folder in pre_pros_data_folder.sub_folders
                    if "test_folder" in folder.folder_name
                ),
                None,
            )
            if test_folder:
                return test_folder.sub_folders
            else:
                print(f"Test folder for dataset '{dataset_name}' not found.")
        else:
            print(f"Dataset '{dataset_name}' not found.")
        return None

    def get_val_df(self, dataset_name):
        dataset = self._find_dataset(dataset_name)
        if dataset:
            pre_pros_data_folder = dataset.sub_folders[1]
            val_folder = next(
                (
                    folder
                    for folder in pre_pros_data_folder.sub_folders
                    if "val_folder" in folder.folder_name
                ),
                None,
            )
            if val_folder:
                return val_folder.sub_folders
            else:
                print(f"Validation folder for dataset '{dataset_name}' not found.")
        else:
            print(f"Dataset '{dataset_name}' not found.")
        return None

    def get_raw_df(self, dataset_name):
        dataset = self._find_dataset(dataset_name)
        if dataset:
            raw_data_folder = dataset.sub_folders[0]
            return raw_data_folder.data.to_dataframe()
        else:
            print(f"Dataset '{dataset_name}' not found.")
        return None

    def get_raw_hash_map_node(self, dataset_name):
        dataset = self._find_dataset(dataset_name)
        if dataset:
            raw_data_folder = dataset.sub_folders[0]
            return raw_data_folder
        else:
            print(f"Dataset '{dataset_name}' not found.")
        return None

    def _balance_classes(self, df):
        labels = df["labels"].unique()
        min_count = min(df["labels"].value_counts())

        balanced_df = pd.DataFrame()
        for label in labels:
            label_data = df[df["labels"] == label].sample(min_count)
            balanced_df = pd.concat([balanced_df, label_data])

        balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)
        return balanced_df

    def _save_data(self):
        with open("dataset_structure.pkl", "wb") as f:
            pickle.dump(self.root, f)
        print("Dataset structure saved.")

    def _load_data(self):
        if os.path.exists("dataset_structure.pkl"):
            with open("dataset_structure.pkl", "rb") as f:
                self.root = pickle.load(f)
            print("Dataset structure loaded.")
        else:
            self.root = Folder("root_folder")
            print("No existing dataset structure found. Initialized new structure.")

    def _find_dataset(self, dataset_name):
        for folder in self.root.sub_folders:
            if folder.folder_name == dataset_name:
                return folder
        return None

    def save(self):
        self._save_data()

    def load(self):
        self._load_data()

    def show_datasets(self):
        if not self.root.sub_folders:
            print("No datasets found.")
            return []

        print("Datasets:")
        for folder in self.root.sub_folders:
            print(folder.folder_name)
        return self.root.sub_folders

    def show_dataset_heads(self, dataset_name):
        if dataset_name not in [obj.folder_name for obj in self.root.sub_folders]:
            print("No such dataset")
            return None, None, None
        train_df = self.get_train_df(dataset_name)
        test_df = self.get_test_df(dataset_name)
        val_df = self.get_val_df(dataset_name)
        x, y, z = None, None, None
        if train_df is not None:
            x = train_df.head()

        if test_df is not None:
            y = test_df.head()

        if val_df is not None:
            z = val_df.head()

        return x, y, z

    def add_images(self, images: list, word_label, dataset_name, balance=True):
        dataset = self._find_dataset(dataset_name)
        if not dataset:
            print("No Such Dataset")
            return None
        hash_map_node = dataset_manager.get_raw_hash_map_node("dataset1")
        class_dict = hash_map_node.word_number_labels
        flag = False
        for label in class_dict.keys():
            if word_label == label:
                flag = True
                break
        if flag == True:
            label = class_dict[word_label]
        else:
            class_dict[word_label] = max(class_dict.values()) + 1

        new_df = pd.DataFrame(
            {"images": images, "labels": [class_dict[word_label]] * len(images)}
        )
        raw_df = dataset_manager.get_raw_df(dataset_name)

        combined_df = pd.concat([raw_df, new_df], axis=0)
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        raw_data = HashMaps()
        raw_data.bulk_insert_from_dataframe(combined_df)
        hash_map_node.data = raw_data

        dataset.sub_folders[0] = hash_map_node

        train_size, test_size, val_size = (
            hash_map_node.train_size,
            hash_map_node.test_size,
            hash_map_node.val_size,
        )

        total_size = len(combined_df)
        train_size = int(total_size * train_size)
        test_size = int(total_size * test_size)
        val_size = int(total_size * val_size)

        train_df = combined_df[:train_size]
        test_df = combined_df[train_size : train_size + test_size]
        val_df = combined_df[train_size + test_size :]

        if balance:
            train_df = self._balance_classes(train_df)
            val_df = self._balance_classes(val_df)
            test_df = self._balance_classes(test_df)

        train_folder = Folder(folder_name="train_folder_" + dataset_name)
        val_folder = Folder(folder_name="val_folder_" + dataset_name)
        test_folder = Folder(folder_name="test_folder_" + dataset_name)

        train_folder.sub_folders = train_df
        test_folder.sub_folders = test_df
        val_folder.sub_folders = val_df

        pre_pros_data_folder = Folder(
            folder_name="pre_pros_data_folder_" + dataset_name
        )
        pre_pros_data_folder.sub_folders = [train_folder, val_folder, test_folder]
        dataset.sub_folders[1] = pre_pros_data_folder

        self._save_data()

        print("Images Succesfully Added")

    def delete_images(self, images: list, dataset_name, balance=True):
        dataset = self._find_dataset(dataset_name)
        if not dataset:
            print("No Such Dataset")
            return None
        hash_map_node = dataset_manager.get_raw_hash_map_node("dataset1")

        confirm = input(
            f"Are you sure you want to delete the images '{dataset_name}'? (y/n): "
        )
        if confirm.lower() == "y":
            for image in images:
                hash_map_node.data.remove(image)
        else:
            return

        dataset.sub_folders[0] = hash_map_node

        train_size, test_size, val_size = (
            hash_map_node.train_size,
            hash_map_node.test_size,
            hash_map_node.val_size,
        )

        combined_df = hash_map_node.data.to_dataframe()

        total_size = len(combined_df)
        train_size = int(total_size * train_size)
        test_size = int(total_size * test_size)
        val_size = int(total_size * val_size)

        train_df = combined_df[:train_size]
        test_df = combined_df[train_size : train_size + test_size]
        val_df = combined_df[train_size + test_size :]

        if balance:
            train_df = self._balance_classes(train_df)
            val_df = self._balance_classes(val_df)
            test_df = self._balance_classes(test_df)

        train_folder = Folder(folder_name="train_folder_" + dataset_name)
        val_folder = Folder(folder_name="val_folder_" + dataset_name)
        test_folder = Folder(folder_name="test_folder_" + dataset_name)

        train_folder.sub_folders = train_df
        test_folder.sub_folders = test_df
        val_folder.sub_folders = val_df

        pre_pros_data_folder = Folder(
            folder_name="pre_pros_data_folder_" + dataset_name
        )
        pre_pros_data_folder.sub_folders = [train_folder, val_folder, test_folder]
        dataset.sub_folders[1] = pre_pros_data_folder

        self._save_data()

        print("Deleted the Images")


dataset_manager = DataManager()
