from pyboy import PyBoy
import threading
from ram_map import *

# Load the ROM file
pyboy = PyBoy('pokemon_red.gb')

def get_memory_value(address):
    return pyboy.get_memory_value(address)

# Function to set the memory value at a specific address
def set_memory_value(address, value):
    pyboy.set_memory_value(address, value)

# Function to display memory values based on user input
import threading


def get_pokeball_count(game):
    # List of Pokeball IDs
    pb_ids = [0x01, 0x02, 0x03, 0x04]
    # Get all items with their quantities from the bag
    items = get_items_in_bag_q(game)
    total_pb_cnt = 0

    # Iterate over the items list two steps at a time to get ID and its quantity
    for i in range(0, len(items), 2):
        item_id = items[i]
        quantity = items[i + 1] if i + 1 < len(items) else 0  # Guard against index out of range

        # If the current item ID is a Pokeball, add its quantity to the total count
        if item_id in pb_ids:
            total_pb_cnt += quantity

    return total_pb_cnt

def get_items_in_bag_q(game, one_indexed=0):
    first_item = 0xD31E
    item_ids = []
    # Assuming each item and its quantity occupy 2 consecutive memory slots
    for i in range(0, 40, 2):  # Adjust the range if there are more than 20 items
        item_id = game.get_memory_value(first_item + i)
        if item_id == 0 or item_id == 0xFF:  # Check if the item slot is empty or end of list marker
            break
        quantity = game.get_memory_value(first_item + i + 1)  # Quantity is next to the ID
        item_ids.extend([item_id + one_indexed, quantity])  # Append both ID and quantity

    return item_ids

def dump_event_flags(pyboy, filename):
    start_address = 0xc022  # Start of event flags region
    end_address = 0xD35C   # End of event flags region
    with open(filename, 'w') as file:
        for addr in range(start_address, end_address + 1):
            value = get_memory_value(addr) 
            file.write(f"{addr:#06X}:{value:02X}\n")
    print(f"Dump saved to {filename}")
    
    
def compare_dumps(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        dump1 = f1.readlines()
        dump2 = f2.readlines()

    changes = {}
    for line1, line2 in zip(dump1, dump2):
        addr1, value1 = line1.strip().split(':')
        addr2, value2 = line2.strip().split(':')
        if value1 != value2:
            changes[addr1] = (value1, value2)

    for addr, (val1, val2) in changes.items():
        print(f"Address {addr} changed from {val1} to {val2}")

def set_memory_bit(address, bit_position, bit_value):
    # Read the current byte value at the address
    current_byte = pyboy.get_memory_value(address)
    bit_position = int(bit_position)
    bit_value = int(bit_value)

    # Modify the bit at bit_position to bit_value
    if bit_value == 1:
        modified_byte = current_byte | (1 << bit_position)
    else:
        modified_byte = current_byte & ~(1 << bit_position)
    
    # Write the modified byte back to the address
    pyboy.set_memory_value(address, modified_byte)

def clear_memory_range(start_address, end_address):
    # Convert start_address and end_address to integers (base 16) if they are strings
    if isinstance(start_address, str):
        start_address = int(start_address, 16)
    if isinstance(end_address, str):
        end_address = int(end_address, 16)
    
    # Iterate over each address in the range
    for address in range(start_address, end_address + 1):
        # Read the current byte value at the address
        current_byte = pyboy.get_memory_value(address)
        
        # Set all bits to 0
        modified_byte = 0
        
        # Write the modified byte back to the address
        pyboy.set_memory_value(address, modified_byte)

def display_memory_values(pyboy):
    while True:
        command = input("Enter a command (eRate, pokemon, common, uncommon, rare, battle, 'hp', 'level', 'money', 'event', 'dump'): ")

        if command == 'hp':
            print("Pokémon HP:", hp(pyboy))
        elif command == 'level':
            print("Pokémon Level:", get_memory_value(PARTY_LEVEL_ADDR[0]))
        elif command == 'money':
            print("Money:", money(pyboy))
        elif command == 'pokemon':
            print("Pokémon:", pokemon(pyboy))
        elif command == 'battle':
            print("In Battle?:", is_in_battle(pyboy))
        elif command == 'eRate':
            encounter_rate = get_wild_encounter_rate(pyboy)
            print("Wild Encounter Rate:", encounter_rate)
        elif command == 'map':
            map = get_memory_value(0xD35E)
            print("Map #:", map)
        elif command == 'pokec':
            pc = get_memory_value(0xD719)
            print("Poke Center #:", pc)
        elif command in ['rare', 'uncommon', 'common']:
            encounters = get_wild_encounters(pyboy, command)
            print(f"{command.capitalize()} Battles:")
            for pokemon, level in encounters:
                print(f"- {pokemon} (Level {level})")
        elif command == 'test':
            testvalue = get_gym_3_lock(pyboy)
            print("get_gym_3_lock test = ", testvalue)
        elif command == 'event':
            addr = int(input("Enter event flag address (hex), e.g., 0xD123: "), 16)
            bit_index = input("Enter bit index (0-7) or press Enter if not applicable: ").strip()
            value = get_memory_value(addr)
            if bit_index:
                bit_index = int(bit_index)
                is_set = (value >> bit_index) & 1
                print(f"Event flag at {addr:#X}, bit {bit_index} is {'set' if is_set else 'not set'}.")
            else:
                print(f"Value at {addr:#X}: {value}")
        elif command == 'dump':
            filename = input("Enter filename to save dump: ")
            dump_event_flags(pyboy, filename)
        elif command == 'compare':
            file1 = input("Enter filename for first dump: ")
            file2 = input("Enter filename for second dump: ")
            compare_dumps(file1, file2)
        elif command == 'read':
            addr = int(input("Enter event flag address (hex), e.g., 0xD123: "), 16)
            bit_index = input("Enter bit index (0-7) or press Enter if not applicable: ").strip()
            value = get_memory_value(addr)
            if bit_index:
                bit_index = int(bit_index)
                is_set = (value >> bit_index) & 1
                print(f"Event flag at {addr:#X}, bit {bit_index} is {'set' if is_set else 'not set'}.")
            else:
                print(f"Value at {addr:#X}: {value}") 
        elif command == 'write':
                addr = int(input("Enter event flag address (hex), e.g., 0xD123: "), 16)
                bit_index = input("Enter bit index (0-7)").strip()
                value = int(input("Enter value").strip(), 16)
                if bit_index:
                    bit_index = int(bit_index)
                    set_memory_bit(addr, bit_index, value) 
                
                else:
                    set_memory_value(addr, value)
                
                      
        elif command == 'wipe':
                addr1 = int(input("start address: "), 16)
                addr2 = input("end address").strip()
                
                clear_memory_range(addr1, addr2)     
        elif command == 'pokeown':
                powned = pokemon_caught(pyboy)
                print(f"Pokemon Owned: {powned}")
        elif command == 'pokeballs':
                pokeballs = get_pokeball_count(pyboy)
                print(f"Pokeballs Owned: {pokeballs}")
        elif command in ['rare_diff', 'uncommon_diff', 'common_diff']:
                battle_type = command.split('_')[0]
                not_owned_names = compare_seen_owned(pyboy, battle_type)
                print(f"Pokémon seen in {battle_type} battles but not owned:")
                print(", ".join(not_owned_names))
                if not not_owned_names:
                    print("You own all the pokemons... Good!")
        else:
            print("Invalid command. Try again.")

def input_thread(pyboy):
    while True:
        user_input = input("Press 'm' to display memory values: ")
        if user_input.lower() == 'm':
            display_memory_values(pyboy)

# Assuming pyboy and other necessary variables and functions are defined elsewhere
# Start the user input thread
input_thread_obj = threading.Thread(target=input_thread, args=(pyboy,))
input_thread_obj.daemon = True
input_thread_obj.start()

# Assuming there's a loop somewhere to keep the script running and call pyboy.tick()


# Start the game loop
while True:
    # Run the game for one frame
    pyboy.tick()
