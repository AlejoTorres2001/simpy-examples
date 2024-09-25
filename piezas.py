import simpy
import numpy as np
import matplotlib.pyplot as plt

# System parameters
NUM_STATIONS = 5
SIMULATION_TIME = 365 * 8  # 365 days of 8 hours each

# Number of machines per station
MACHINES_PER_STATION = [3, 2, 4, 4, 1]
#MACHINES_PER_STATION = [3, 2, 4, 4, 1]  # Added one machine to Station 4
# Mean service times (in hours) per piece type and station
SERVICE_TIMES = {
    1: [0.85, 0.5, 0.6, 0.5],  # Type 1: stations 3, 1, 2, 5
    2: [1.1, 0.8, 0.75],        # Type 2: stations 4, 1, 3
    3: [1.2, 0.25, 0.7, 0.9, 1.0]  # Type 3: stations 2, 5, 1, 4, 3
}

# Routes by piece type
ROUTES = {
    1: [2, 0, 1, 4],  # Type 1: stations 3, 1, 2, 5
    2: [3, 0, 2],     # Type 2: stations 4, 1, 3
    3: [1, 4, 0, 3, 2] # Type 3: stations 2, 5, 1, 4, 3
}

# Probabilities of piece types
PIECE_PROBABILITIES = [0.3, 0.5, 0.2]

# Exponential distribution for arrivals
ARRIVAL_MEAN = 0.25

# Create workstation
class Station:
    def __init__(self, env, num_machines, station_id):
        self.env = env
        self.station_id = station_id
        self.machines = simpy.Resource(env, num_machines)
        self.total_wait_time = 0
        self.total_busy_time = 0
        self.num_pieces = 0
        self.wait_times = []

    def process(self, piece_id, piece_type, service_time):
        with self.machines.request() as request:
            yield request
            # Simulate processing time
            process_time = np.random.exponential(service_time)
            self.total_busy_time += process_time  # Accumulate busy time
            yield self.env.timeout(process_time)
            self.num_pieces += 1

    def add_wait_time(self, wait_time):
        self.total_wait_time += wait_time
        self.wait_times.append(wait_time)

# Create piece
class Piece:
    def __init__(self, env, id, type, system):
        self.env = env
        self.id = id
        self.type = type
        self.system = system
        self.start_time = env.now  # Arrival time
        self.wait_times = []
        self.processes_completed = 0
        self.action = env.process(self.process())

    def process(self):
        route = ROUTES[self.type]
        service_times = SERVICE_TIMES[self.type]
        for i, station_id in enumerate(route):
            station = self.system.stations[station_id]
            service_time = service_times[i]

            # Log arrival time and queue wait time
            wait_start_time = self.env.now
            yield self.env.process(station.process(self.id, self.type, service_time))
            wait_time = self.env.now - wait_start_time

            # Update wait time metrics
            station.add_wait_time(wait_time)
            self.wait_times.append(wait_time)
            self.processes_completed += 1

        # Calculate total cycle time
        cycle_time = self.env.now - self.start_time
        self.system.cycle_times.append(cycle_time)

# Manufacturing system
class ManufacturingSystem:
    def __init__(self, env):
        self.env = env
        self.stations = [Station(env, MACHINES_PER_STATION[i], i) for i in range(NUM_STATIONS)]
        self.num_pieces_created = 0
        self.cycle_times = []  # To log cycle time of each piece

    def generate_piece(self):
        while True:
            piece_type = np.random.choice([1, 2, 3], p=PIECE_PROBABILITIES)
            self.num_pieces_created += 1
            Piece(self.env, self.num_pieces_created, piece_type, self)
            # Time between arrivals
            yield self.env.timeout(np.random.exponential(ARRIVAL_MEAN))

    def run_simulation(self):
        # Start the piece generation process
        self.env.process(self.generate_piece())
        # Run the simulation for the defined time
        self.env.run(until=SIMULATION_TIME)
        # Calculate results
        self.calculate_statistics()

    def calculate_statistics(self):
        print(f"Simulation completed. Processed pieces: {self.num_pieces_created}")
        utilizations = []
        avg_wait_times = []
        avg_pieces_in_queue = []

        for i, station in enumerate(self.stations):
            utilization = station.total_busy_time / (SIMULATION_TIME * MACHINES_PER_STATION[i])
            utilizations.append(utilization)
            avg_wait_time = station.total_wait_time / station.num_pieces if station.num_pieces > 0 else 0
            avg_wait_times.append(avg_wait_time)
            avg_pieces_in_queue.append(len(station.wait_times) / SIMULATION_TIME)

            print(f"Station {i+1}:")
            print(f"  Utilization: {utilization}")
            print(f"  Average wait time: {avg_wait_time}")
            print(f"  Processed pieces: {station.num_pieces}")

        # Calculate arrival and departure rates
        arrival_rate = self.num_pieces_created / SIMULATION_TIME
        departure_rate = sum([station.num_pieces for station in self.stations]) / SIMULATION_TIME
        print(f"Arrival rate: {arrival_rate}")
        print(f"Departure rate: {departure_rate}")
        print(f"Average cycle time: {np.mean(self.cycle_times)}")

        print("\nFinal analysis completed.\n")

        # Plot metrics
        self.plot_metrics(utilizations, avg_wait_times, avg_pieces_in_queue)

    def plot_metrics(self, utilizations, avg_wait_times, avg_pieces_in_queue):
        stations = [f"Station {i+1}" for i in range(NUM_STATIONS)]

        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Utilization plot
        axs[0, 0].bar(stations, utilizations, color='skyblue')
        axs[0, 0].set_title('Utilization of each station')
        axs[0, 0].set_xlabel('Stations')
        axs[0, 0].set_ylabel('Utilization')

        # Average wait time plot
        axs[0, 1].bar(stations, avg_wait_times, color='lightgreen')
        axs[0, 1].set_title('Average wait time per station')
        axs[0, 1].set_xlabel('Stations')
        axs[0, 1].set_ylabel('Average wait time (hours)')

        # Average pieces in queue plot
        axs[1, 0].bar(stations, avg_pieces_in_queue, color='salmon')
        axs[1, 0].set_title('Average number of pieces in queue per station')
        axs[1, 0].set_xlabel('Stations')
        axs[1, 0].set_ylabel('Average pieces in queue')

        # Cycle time distribution
        axs[1, 1].hist(self.cycle_times, bins=20, color='orchid')
        axs[1, 1].set_title('Cycle time distribution')
        axs[1, 1].set_xlabel('Cycle time (hours)')
        axs[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()


# Run the simulation
env = simpy.Environment()
system = ManufacturingSystem(env)
system.run_simulation()
