import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

# Parameters
RANDOM_SEED = 42
OPERATING_HOURS = 8 * 60  # The bank operates for 8 hours -> minutes
ARRIVAL_RATE = 1  # Mean inter-arrival time, in minutes (exponential distribution)
SERVICE_RATE = 4.5  # Mean service time, in minutes (exponential distribution)

metrics = {
    'wait_times': [],           # Average wait times over time (recorded each minute)
    'service_times': [],        # Service times per client
    'teller_utilization': [],   # Utilization per teller over time
    'queue_lengths': [],        # Queue lengths over time
    'clients_served': 0,       # Total clients served (single value)
    'jockeying_events': []      # Cumulative jockeying events over time
}

# Global clock
simulation_time = []

class Client:
    def __init__(self, env, client_id, bank):
        self.env = env
        self.client_id = client_id
        self.bank = bank
        self.arrival = self.env.now
        self.delay = 0
        env.process(self.client_process())

    def client_process(self):
        # The client selects the shortest queue
        teller = self.bank.select_teller()
        # If the teller is busy, the client will wait in the queue
        with teller.request() as req:
            yield req
            # Client begins service
            self.delay = self.env.now - self.arrival
            teller.record_delay(self.delay)
            service_time = random.expovariate(1 / SERVICE_RATE)
            yield self.env.timeout(service_time)
            metrics['service_times'].append(service_time)
            
            teller.time_busy += service_time

            self.bank.check_jockeying()

class Teller:
    def __init__(self, env, teller_id):
        self.env = env
        self.teller_id = teller_id
        self.queue = simpy.Resource(env, capacity=1)
        self.delays = []
        self.time_busy = 0

    def request(self):
        return self.queue.request()

    def record_delay(self, delay):
        self.delays.append(delay)

    def average_delay(self):
        return np.mean(self.delays) if self.delays else 0

    def maximum_delay(self):
        return np.max(self.delays) if self.delays else 0

    def utilization(self, current_time):
        return self.time_busy / current_time if current_time > 0 else 0

class Bank:
    def __init__(self, env, num_tellers):
        self.env = env
        self.num_tellers = num_tellers
        self.tellers = [Teller(env, i) for i in range(num_tellers)]
        self.total_delays = []
        self.jockeying_events = 0

    def select_teller(self):
        # Choose the teller with the shortest queue
        teller = min(self.tellers, key=lambda tel: len(tel.queue.queue))
        return teller

    def check_jockeying(self):
        # Check if any client should switch queues
        for i, teller_i in enumerate(self.tellers):
            for j, teller_j in enumerate(self.tellers):
                if len(teller_j.queue.queue) > len(teller_i.queue.queue) + 1:
                    # Try to move a client from the longer queue to the shorter one
                    if teller_j.queue.count > 0:
                        self.jockeying_events += 1  # Register jockeying event

def generate_clients(env, bank, arrival_rate):
    client_id = 0
    while True:
        yield env.timeout(random.expovariate(1 / arrival_rate))
        client_id += 1
        Client(env, client_id, bank)

def track_queue_lengths(env, bank):
    while True:
        queue_length = sum(len(teller.queue.queue) for teller in bank.tellers)
        metrics['queue_lengths'].append(queue_length)
        simulation_time.append(env.now)

        all_wait_times = [teller.average_delay() for teller in bank.tellers if teller.delays]
        avg_wait_time = np.mean(all_wait_times) if all_wait_times else 0
        metrics['wait_times'].append(avg_wait_time)

        metrics['jockeying_events'].append(bank.jockeying_events)

        yield env.timeout(1)  # Track every minute

def run_simulation(num_tellers):
    random.seed(RANDOM_SEED)
    env = simpy.Environment()
    bank = Bank(env, num_tellers)
    env.process(generate_clients(env, bank, ARRIVAL_RATE))
    env.process(track_queue_lengths(env, bank))
    env.run(until=OPERATING_HOURS)

    metrics['clients_served'] = len(metrics['service_times'])

    return bank  # Return the bank object for plotting

def plot_metrics(bank):
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))  # Create a 3x2 grid for subplots
    fig.suptitle('Bank Simulation Metrics', fontsize=16)

    # 1. Plot average wait times
    axs[0, 0].plot(simulation_time, metrics['wait_times'], label="Average Wait Time", color='blue')
    axs[0, 0].set_xlabel("Time (minutes)")
    axs[0, 0].set_ylabel("Wait Time (minutes)")
    axs[0, 0].set_title("Average Wait Time Over Time")
    axs[0, 0].legend()

    # 2. Plot service times
    axs[0, 1].plot(simulation_time[:len(metrics['service_times'])], metrics['service_times'], label="Service Time", color='orange')
    axs[0, 1].set_xlabel("Time (minutes)")
    axs[0, 1].set_ylabel("Service Time (minutes)")
    axs[0, 1].set_title("Service Time Over Time")
    axs[0, 1].legend()

    # 3. Plot teller utilization
    for teller in bank.tellers:
        axs[1, 0].plot(simulation_time, [teller.utilization(t) for t in simulation_time], label=f"Teller {teller.teller_id}")
    axs[1, 0].set_xlabel("Time (minutes)")
    axs[1, 0].set_ylabel("Utilization")
    axs[1, 0].set_title("Teller Utilization Over Time")
    axs[1, 0].legend()

    # 4. Plot queue lengths over time
    axs[1, 1].plot(simulation_time, metrics['queue_lengths'], label="Queue Length", color='green')
    axs[1, 1].set_xlabel("Time (minutes)")
    axs[1, 1].set_ylabel("Queue Length (Number of Clients)")
    axs[1, 1].set_title("Queue Length Over Time")
    axs[1, 1].legend()

    # 5. Plot total clients served over time
    cumulative_clients = np.arange(1, metrics['clients_served'] + 1)
    axs[2, 0].plot(simulation_time[:len(cumulative_clients)], cumulative_clients, label="Clients Served", color='purple')
    axs[2, 0].set_xlabel("Time (minutes)")
    axs[2, 0].set_ylabel("Cumulative Clients Served")
    axs[2, 0].set_title("Cumulative Clients Served Over Time")
    axs[2, 0].legend()

    # 6. Plot jockeying events over time
    axs[2, 1].plot(simulation_time, metrics['jockeying_events'], label="Jockeying Events", color='red')
    axs[2, 1].set_xlabel("Time (minutes)")
    axs[2, 1].set_ylabel("Cumulative Jockeying Events")
    axs[2, 1].set_title("Cumulative Jockeying Events Over Time")
    axs[2, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the main title
    plt.show()

# Run the simulation and plot the metrics
if __name__ == '__main__':
    bank = run_simulation(6)
    plot_metrics(bank)
