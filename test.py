# Given the equation K = p²/2m
# Let's define a function to calculate the kinetic energy first.

def calculate_kinetic_energy(mass, momentum):
    # kinetic energy = p^2 / (2m)
    return (momentum ** 2) / (2 * mass)

# Now we will define a function to calculate the percentage increase in kinetic energy
# when momentum is increased by a certain percentage.

def calculate_K_increase(mass, momentum, percent_increase):
    # Calculate original kinetic energy
    original_K = calculate_kinetic_energy(mass, momentum)
    
    # Calculate new momentum after increase
    new_momentum = momentum * (1 + percent_increase / 100)
    
    # Calculate new kinetic energy with increased momentum
    new_K = calculate_kinetic_energy(mass, new_momentum)
    
    # Calculate percentage increase in kinetic energy
    K_increase = ((new_K - original_K) / original_K) * 100
    
    return K_increase

# Now let's validate this script with a range of different masses.
# We will take a fixed momentum and see how a 20% increase in momentum affects the kinetic energy
# for different masses.

initial_momentum = 10
percent_increase = 20
mass_range = range(1, 11)  # 1 to 10

percentage_increases = []

for mass in mass_range:
    increase = calculate_K_increase(mass, initial_momentum, percent_increase)
    percentage_increases.append((mass, increase))

print(percentage_increases)