
# Activity Dictionary is manually typed out for clarity, the rationale being the Activity dictionary 
# provided lacks clarification on what the activities are.

activity_dict = {'A': 'Walking', 'B': 'Jogging', 'C': 'Stairs', 'D': 'Sitting', 'E': 'Standing',
                 'F': 'Typing', 'G': 'Brushing teeth', 'H': 'Eating soup', 'I': 'Eating chips',
                 'J': 'Eating pasta', 'K': 'Drinking from cup', 'L': 'Eating sandwich',
                 'M': 'Kicking something', 'O': 'Playing catch with ball', 
                 'P': 'Dribbling basketball', 'Q': 'Writing', 'R': 'Clapping',
                 'S': 'Folding clothes'}

def get_activity_name(key):
    print(key)
    return activity_dict[key]

def return_dict():
    return sorted(activity_dict)


 