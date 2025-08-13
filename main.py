from wiki import get_page, find_short_path
import random
import warnings
import nltk
import spacy

# Suppress HTML parser warnings from wikipedia library and bs4
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")
warnings.filterwarnings("ignore", category=UserWarning, module="bs4")
warnings.filterwarnings("ignore", message=".*parser.*", category=UserWarning)

def main():
    print("\n\n🥓 Welcome to WikiBacon! 🥓\n")
    print("In this game, we start from a random Wikipedia page, and then we compete to see who can name a page that is *farthest away* from the original page.\n")
    
    print("Choose game mode:")
    print("1. Normal mode (uses both page links and categories)")
    print("2. Hard mode (uses only page links, no categories)")
    mode_choice = input("Enter 1 or 2: ").strip()
    
    hard_mode = mode_choice == "2" # So if you enter anything that is not "2", it will default to normal mode.
    if hard_mode:
        print("\n🥓 Hard mode enabled! Categories are disabled for more challenging gameplay. 🥓\n")
    else:
        print("\n🥓 Normal mode enabled! 🥓\n")
    
    print("Ready to play? Hit Enter to start, or type 'q' to quit")
    cmd = input()
    if cmd == "q":
        return
    
    with open("dictionary.txt", "r") as f:
        common_words = f.read().splitlines()

    while True:
        # Get a valid start page
        start_page = None
        attempts = 0
        while start_page is None and attempts < 10:
            start_word = random.choice(common_words)
            start_page = get_page(start_word)
            attempts += 1
        
        if start_page is None: # If we couldn't find a valid starting page, try again. Chances of this happening are very low.
            print("Could not find a valid starting page. Please try again.")
            continue
            
        print(f"The starting page is: {start_page.title}\n")
        print(f"Summary: {start_page.summary[:500]}...\n")

        # Get a valid computer page
        computer_page = None
        attempts = 0
        while computer_page is None and attempts < 10:
            computer_word = random.choice(common_words)
            computer_page = get_page(computer_word)
            attempts += 1
        
        if computer_page is None: # If we couldn't find a valid starting page, try again. Chances of this happening are very low.
            print("Could not find a valid computer page. Please try again.")
            continue

        print(f"The computer's page is: {computer_page.title}\n")
        print(f"Summary: {computer_page.summary[:500]}...\n")

        # Get user page
        print("What would you like your page to be?")
        user_page_name = input()
        user_page = get_page(user_page_name)
        
        if user_page is None:
            print("Could not find that page. Please try again.")
            continue
            
        print(f"Your page is: {user_page.title}\n")
        print(f"Summary: {user_page.summary[:500]}...\n")

        print("Calculating Bacon paths...\n")

        computer_path = find_short_path(start_page, computer_page, hard_mode)
        print("Computer's path:")
        if computer_path[0].startswith("No path found") or computer_path[0].startswith("Error:"):
            print(computer_path[0])
            computer_length = 0
        else:
            print(f"\n -> ".join(computer_path))
            computer_length = len(computer_path)
        print(f"Length: {computer_length}\n")

        user_path = find_short_path(start_page, user_page, hard_mode)
        print("Your path:")
        if user_path[0].startswith("No path found") or user_path[0].startswith("Error:"):
            print(user_path[0])
            user_length = 0
        else:
            print(f"\n -> ".join(user_path))
            user_length = len(user_path)
        print(f"Length: {user_length}\n")

        if computer_length > user_length:
            print("I win!")
        elif computer_length < user_length:
            print("You win!")
        else:
            print("It's a tie!")

        print("\n\nPlay again? Hit Enter for another round, or type 'q' to quit")
        cmd = input()
        if cmd == "q":
            print("\n🥓 Thanks for playing! 🥓\n")
            print("WikiBacon is not affiliated with Wikipedia or the Wikimedia Foundation. To donate to Wikipedia and support their vision of an open internet that makes games like this possible, please visit https://donate.wikimedia.org/\n")
            return

if __name__ == "__main__":
    main()