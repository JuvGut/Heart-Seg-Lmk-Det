import random

fortunes = [
    "A pleasant surprise is waiting for you.",
    "Adventure awaits around the next corner!",
    "Your creativity will lead you to great success.",
    "Good fortune will be yours.",
    "A smile is your passport into the hearts of others.",
    "Your hard work will pay off soon.",
    "A thrilling time is in your near future.",
    "You will find happiness in unexpected places.",
    "Your kindness will lead you to great places.",
    "A new friendship will bring you joy."
]

def get_fortune():
    return random.choice(fortunes)

print("Welcome to the Fortune Cookie Generator!")
input("Press Enter to crack open your fortune cookie...")

print("\nYour fortune says:")
print(f">>> {get_fortune()} <<<")

print("\nThanks for using the Fortune Cookie Generator!")
