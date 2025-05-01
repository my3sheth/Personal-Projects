import pandas as pd
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt

class StudentRecord:
    def __init__(self, class_name):
        self.class_name = class_name
        self.subjects = []
        self.marks = []

    def input_subjects_and_marks(self):
        print(f"To terminate the input of subjects for {self.class_name}, type 'end'")
        s = 1
        while True:
            subject = input(f'Enter name of Subject {s}: ')
            s += 1
            if subject.lower() == 'end':
                break
            marks = int(input(f'Enter marks scored in {subject}: '))
            self.subjects.append(subject)
            self.marks.append(marks)

    def generate_report(self):
        df = pd.DataFrame({'SUBJECTS': self.subjects, 'MARKS': self.marks}, index=range(1, len(self.subjects) + 1))
        grade = pd.cut(df['MARKS'], bins=[0, 40, 60, 70, 80, 100], labels=['E', 'D', 'C', 'B', 'A'])
        df['GRADE'] = grade
        total_marks = np.sum(df['MARKS'])
        total_subjects = len(self.subjects)
        percentage = int(total_marks / (100 * total_subjects) * 100)

        print('\n', df, '\n')
        print(f'TOTAL: {total_marks} / {100 * total_subjects}')
        print(f'PERCENTAGE: {percentage}%')

        return df, percentage

    def visualize_marks(self):
        df, _ = self.generate_report()
        plt.figure(figsize=(8, 6))
        sns.barplot(data=df, x='SUBJECTS', y='MARKS', palette='rocket')
        plt.title(f'{self.class_name} Marks')
        plt.xlabel('Subjects')
        plt.ylabel('Marks')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

class RecordsManager:
    def __init__(self):
        self.records = {}

    def add_record(self, record):
        self.records[record.class_name] = record

    def visualize_class(self, class_name):
        if class_name in self.records:
            self.records[class_name].visualize_marks()
        else:
            print(f"No data found for {class_name}")

    def compare_classes(self):
        data = {class_name: record.generate_report()[1] for class_name, record in self.records.items()}
        sns.lineplot(data=pd.Series(data), marker='o', markersize=8, color='black')
        plt.title('COMPARISON BETWEEN CLASSES')
        plt.xlabel('CLASS')
        plt.ylabel('PERCENTAGE')
        plt.show()

# Create instances for each class
records_manager = RecordsManager()

class_10 = StudentRecord('CLASS 10')
class_10.input_subjects_and_marks()
records_manager.add_record(class_10)

class_11 = StudentRecord('CLASS 11')
class_11.input_subjects_and_marks()
records_manager.add_record(class_11)

class_12 = StudentRecord('CLASS 12')
class_12.input_subjects_and_marks()
records_manager.add_record(class_12)

# Visualization
print("\nSelect the module to be visualized from the table of contents below:")
print(' 1. 10th Marks \n 2. 11th Marks \n 3. 12th Marks \n 4. Comparison between classes \n 5. Exit \n')

while True:
    choice = int(input('Enter choice of visualization to view: '))
    if choice == 1:
        records_manager.visualize_class('CLASS 10')
    elif choice == 2:
        records_manager.visualize_class('CLASS 11')
    elif choice == 3:
        records_manager.visualize_class('CLASS 12')
    elif choice == 4:
        records_manager.compare_classes()
    elif choice == 5:
        break

print('\nThank you, hope you had an interactive session!')
