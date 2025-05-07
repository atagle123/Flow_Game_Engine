class FormatMazeData: 
    def __init__(self, maze, goal):
        self.maze = maze
        self.goal = goal 
        self.data = []
    
    def add_transitions(self, transitions):
        