import pygame
import random
import tensorflow as tf
import numpy as np

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Set up the game window
pygame.init()
screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption("Simulation")

# Set up the turtle sprite
class Turtle(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([50, 50])
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self):
        pass



# Set up the objective sprite
class Objective(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface([50, 50])
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def update(self):
        pass

# Create a group of sprites
all_sprites = pygame.sprite.Group()
turtle = Turtle(250, 250)
all_sprites.add(turtle)
objective = Objective(100, 100)
all_sprites.add(objective)

# Define the neural network
input_size = 4
hidden_size = 10
output_size = 2

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_size, input_dim=input_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='linear')
])

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Set up the game clock
clock = pygame.time.Clock()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get the turtle's position and the objective's position
    turtle_x, turtle_y = turtle.rect.x, turtle.rect.y
    objective_x, objective_y = objective.rect.x, objective.rect.y

    # Feed the turtle's position and the objective's position into the neural network
    input_data = np.array([turtle_x, turtle_y, objective_x, objective_y]).reshape(1, 4)
    output_data = model.predict(input_data)

    # Update the turtle's position based on the neural network's output
    x_new = turtle.rect.x + int(output_data[0][0])
    y_new = turtle.rect.y + int(output_data[0][1])

    # Limit the turtle's movement to the range (0, 500) in both x and y directions
    x_new = max(0, min(x_new, 500))
    y_new = max(0, min(y_new, 500))

    turtle.rect.x = x_new
    turtle.rect.y = y_new
    

    # Fill the screen with black
    screen.fill(BLACK)

    # Draw the sprites
    all_sprites.draw(screen)

    # Update the display
    pygame.display.update()

    # Calculate the distance between the turtle and the objective
    distance = ((turtle_x - objective_x) ** 2 + (turtle_y - objective_y) ** 2) ** 0.5

    # If the turtle reaches the objective, reset the objective's position
    if distance < 50:
        objective.rect.x = random.randint(50, 450)
        objective.rect.y = random.randint(50, 450)

    # Train the neural network to move towards the objective
    target = np.array([objective_x - turtle_x, objective_y - turtle_y]).reshape(1, 2)
    model.fit(input_data, target, epochs=1, verbose=0)

    # Tick the clock
    clock.tick(60)

# Quit the game
pygame.quit()
