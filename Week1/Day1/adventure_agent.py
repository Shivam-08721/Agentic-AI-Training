import re

class AdventureAgent:
    """
    A simple agent that processes commands for a text-based adventure game.
    This simulates an AI game agent without using any actual AI.
    """
    
    def __init__(self):
        # Define the game world - a simple 2x2 grid
        self.world = {
            (0, 0): {
                "description": "You are in a small clearing in the forest. There's a path to the north and east.",
                "items": ["stick"]
            },
            (0, 1): {
                "description": "You've reached a bubbling stream. The water looks clear. There's a path to the south and east.",
                "items": ["water flask"]
            },
            (1, 0): {
                "description": "You're at the entrance of a small cave. It looks dark inside. There's a path to the west and north.",
                "items": ["rock"]
            },
            (1, 1): {
                "description": "You're on top of a small hill. You can see the forest stretching out around you. There's a path to the west and south.",
                "items": ["map"]
            }
        }
        
        # Player's current position
        self.position = (0, 0)
        
        # Player's inventory
        self.inventory = []
        
        # Game state
        self.game_over = False
    
    def process_command(self, command):
        """Process user command and return an appropriate response"""
        command = command.lower().strip()
        
        # Movement commands
        if re.match(r"go (north|south|east|west)", command) or command in ["north", "south", "east", "west"]:
            direction = command.split()[-1]  # Get the last word (direction)
            return self.move(direction)
        
        # Look command
        elif command == "look":
            return self.look()
        
        # Inventory command
        elif command == "inventory" or command == "i":
            return self.check_inventory()
        
        # Take item command
        elif command.startswith("take ") or command.startswith("get "):
            item = command.split(" ", 1)[1]  # Get everything after 'take' or 'get'
            return self.take_item(item)
        
        # Drop item command
        elif command.startswith("drop "):
            item = command.split(" ", 1)[1]  # Get everything after 'drop'
            return self.drop_item(item)
        
        # Use item command
        elif command.startswith("use "):
            item = command.split(" ", 1)[1]  # Get everything after 'use'
            return self.use_item(item)
        
        # Help command
        elif command == "help":
            return """Available commands:
- 'go north/south/east/west' or simply 'north/south/east/west' - Move in that direction
- 'look' - Look around your current location
- 'inventory' or 'i' - Check what you're carrying
- 'take [item]' or 'get [item]' - Pick up an item
- 'drop [item]' - Drop an item from your inventory
- 'use [item]' - Use an item from your inventory
- 'quit' - Exit the game"""
        
        # Handle unknown commands
        else:
            return "I don't understand that command. Type 'help' for a list of available commands."
    
    def move(self, direction):
        """Move player in specified direction"""
        x, y = self.position
        
        if direction == "north":
            new_position = (x, y + 1)
        elif direction == "south":
            new_position = (x, y - 1)
        elif direction == "east":
            new_position = (x + 1, y)
        elif direction == "west":
            new_position = (x - 1, y)
        else:
            return "That's not a valid direction."
        
        # Check if the new position is valid
        if new_position in self.world:
            self.position = new_position
            return self.look()
        else:
            return "You can't go that way."
    
    def look(self):
        """Look around current location"""
        location = self.world[self.position]
        description = location["description"]
        
        # Add information about items at the location
        items = location["items"]
        if items:
            item_list = ", ".join(items)
            description += f"\nYou can see: {item_list}"
        
        return description
    
    def check_inventory(self):
        """Show player's inventory"""
        if not self.inventory:
            return "Your inventory is empty."
        else:
            item_list = ", ".join(self.inventory)
            return f"You are carrying: {item_list}"
    
    def take_item(self, item):
        """Pick up an item from current location"""
        location = self.world[self.position]
        
        if item in location["items"]:
            location["items"].remove(item)
            self.inventory.append(item)
            return f"You take the {item}."
        else:
            return f"There is no {item} here."
    
    def drop_item(self, item):
        """Drop an item from inventory"""
        if item in self.inventory:
            self.inventory.remove(item)
            self.world[self.position]["items"].append(item)
            return f"You drop the {item}."
        else:
            return f"You don't have a {item}."
    
    def use_item(self, item):
        """Use an item from inventory"""
        if item not in self.inventory:
            return f"You don't have a {item}."
        
        # Different outcomes based on item and location
        if item == "water flask" and self.position == (0, 1):
            return "You fill your water flask with fresh water from the stream."
        elif item == "map":
            return "The map shows your current position in the forest and marks a mysterious 'X' somewhere to the north."
        elif item == "stick" and self.position == (1, 0):
            return "You poke around in the cave entrance with the stick, disturbing some bats that fly out."
        elif item == "rock" and self.position == (1, 1):
            return "You place the rock on top of the hill, marking your visit."
        else:
            return f"You're not sure how to use the {item} here."

def main():
    agent = AdventureAgent()
    print("Welcome to the Simple Text Adventure!")
    print("Type 'help' for a list of commands or 'quit' to exit.")
    print(agent.look())  # Show initial location
    
    while not agent.game_over:
        command = input("> ")
        
        if command.lower() == "quit":
            print("Thanks for playing!")
            break
        
        response = agent.process_command(command)
        print(response)

if __name__ == "__main__":
    main()
