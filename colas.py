from random import seed
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

class Client:
    @staticmethod
    def generate_clients(num_clients=10, arrival_dist=None, service_dist=None):
        """
        Generate a list of clients with specified arrival and service time distributions.
        """
        clients = []
        arrival_time = 0
        for i in range(1, num_clients + 1):
            if arrival_dist is not None:
                inter_arrival = max(1, int(arrival_dist()))  # Ensure at least 1 time unit between arrivals
                arrival_time += inter_arrival
            if service_dist is not None:
                service_time = max(1, int(service_dist()))  # Ensure service time is at least 1
            else:
                service_time = np.random.randint(1, 10)
            clients.append(Client(i, arrival_time, service_time))
        return clients

    def __init__(self, id: int, arrival_time: int, service_time: int):
        self.id = id
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.original_service_time = service_time
        
        self.time_in_system = 0
        self.time_in_queue = 0
        self.time_service_begins = 0
        self.time_service_end = 0
        self.server:Optional[Server] = None

    def __str__(self):
        return (f'Client {self.id} arrived at {self.arrival_time}, '
                f'service time {self.original_service_time} minutes. '
                f'Queue time {self.time_in_queue}, system time {self.time_in_system}. '
                f'Service on server {self.server.id if self.server else "None"}')

class Server:
    @staticmethod
    def generate_servers(num: int = 2):
        """
        Generate a list of servers.
        """
        return [Server(idx + 1) for idx in range(num)]

    def __init__(self, id: int):
        self.id = id
        self.is_busy = False
        self.current_client:Optional[Client] = None
        self.attended_clients:list = []

    def assign_client(self, client: Client, current_time: int):
        """
        Assign a client to this server.
        """
        self.current_client = client
        self.is_busy = True
        client.server = self
        client.time_service_begins = current_time
        client.time_in_queue = current_time - client.arrival_time

    def release_client(self, current_time: int):
        """
        Release the current client from this server.
        """
        if self.current_client is not None:
            self.current_client.time_service_end = current_time
            self.current_client.time_in_system = self.current_client.time_service_end - self.current_client.arrival_time
        self.attended_clients.append(self.current_client)
        self.is_busy = False
        self.current_client = None

class Simulation:
    def __init__(self, clients: list[Client], servers: list[Server]):
        self.current_time = 0
        self.servers = servers
        # Sort clients by arrival time
        self.clients = sorted(clients, key=lambda x: x.arrival_time)
        self.waiting_queue:list = []  # Separate waiting queue
        self.queue_lengths:list = []
        self.times:list = []

    def run(self):
        """
        Run the simulation until all clients are processed.
        """
        while self.clients or self.waiting_queue or any(server.is_busy for server in self.servers):
            # Assign arriving clients to servers or queue
            while self.clients and self.clients[0].arrival_time <= self.current_time:
                client = self.clients.pop(0)
                # Find a free server
                free_server = next((s for s in self.servers if not s.is_busy), None)
                if free_server:
                    free_server.assign_client(client, self.current_time)
                else:
                    self.waiting_queue.append(client)

            # Serve clients
            for server in self.servers:
                if server.is_busy:
                    server.current_client.service_time -= 1
                    if server.current_client.service_time <= 0:
                        server.release_client(self.current_time)

            # Assign waiting queue to free servers
            while self.waiting_queue and any(not s.is_busy for s in self.servers):
                free_server = next((s for s in self.servers if not s.is_busy), None)
                if free_server:
                    client = self.waiting_queue.pop(0)
                    free_server.assign_client(client, self.current_time)
                else:
                    break

            # Track queue length for plotting
            queue_length = len(self.waiting_queue)
            self.queue_lengths.append(queue_length)
            self.times.append(self.current_time)

            self.current_time += 1

            # Optional: Remove the safeguard if the simulation completes correctly
            if self.current_time > 10000:
                print("Simulation exceeded time limit. Stopping.")
                break

    def plot(self):
        """
        Print details of each attended client.
        """
        print("\n--- Simulation Results ---")
        for client in sorted([client for server in self.servers for client in server.attended_clients], key=lambda x: x.id):
            print(client)

    def plot_metrics(self):
        """
        Plot additional metrics for system performance: queue length, system time per client, 
        waiting time distribution, server utilization, service time distribution, and time in queue vs system.
        """
        plt.figure(figsize=(16, 12))
        
        # Plot queue length over time
        plt.subplot(2, 3, 1)
        plt.plot(self.times, self.queue_lengths, label='Queue Length', color='blue')
        plt.xlabel('Time')
        plt.ylabel('Number of Clients in Queue')
        plt.title('Queue Length Over Time')
        plt.grid(True)
        plt.legend()
        
        # Plot system time per client
        plt.subplot(2, 3, 2)
        clients = sorted([client for server in self.servers for client in server.attended_clients], key=lambda x: x.id)
        client_ids = [client.id for client in clients]
        system_times = [client.time_in_system for client in clients]

        plt.bar(client_ids, system_times, color='green')
        plt.xlabel('Client')  # Updated to say "Client"
        plt.ylabel('Time in System')
        plt.title('Client System Time')
        plt.xticks(client_ids)  # Ensure that only integer client IDs are shown
        plt.grid(True)

        # Plot waiting time distribution
        waiting_times = [client.time_in_queue for client in clients]
        plt.subplot(2, 3, 3)
        plt.hist(waiting_times, bins=10, color='orange', edgecolor='black')
        plt.xlabel('Time in Queue')
        plt.ylabel('Number of Clients')
        plt.title('Waiting Time Distribution')
        plt.grid(True)

        # Plot server utilization
        server_utilization = [sum(client.time_in_system for client in server.attended_clients) / self.current_time for server in self.servers]
        server_ids = [server.id for server in self.servers]
        plt.subplot(2, 3, 4)
        plt.bar(server_ids, server_utilization, color='purple')
        plt.xlabel('Server')
        plt.ylabel('Utilization (%)')
        plt.title('Server Utilization')
        plt.grid(True)

        # Plot service time distribution
        service_times = [client.original_service_time for client in clients]
        plt.subplot(2, 3, 5)
        plt.hist(service_times, bins=10, color='red', edgecolor='black')
        plt.xlabel('Service Time')
        plt.ylabel('Number of Clients')
        plt.title('Service Time Distribution')
        plt.grid(True)

        # Plot time in queue vs time in system
        plt.subplot(2, 3, 6)
        plt.scatter(waiting_times, system_times, color='brown')
        plt.xlabel('Time in Queue')
        plt.ylabel('Time in System')
        plt.title('Time in Queue vs Time in System')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

       

if __name__ == '__main__':
    seed(42)
    np.random.seed(42)
    
    service_dist = lambda: max(1, np.random.normal(loc=4, scale=1.5))  # Normal distribution for service time, mean=4, std=1.5
    arrival_dist = lambda: np.random.exponential(scale=2)  # Mean inter-arrival time of 2
    
    # Generate clients and servers
    clients = Client.generate_clients(num_clients=10, arrival_dist=arrival_dist, service_dist=service_dist)
    servers = Server.generate_servers(2)
    
    # Run simulation
    sim = Simulation(clients=clients, servers=servers)
    sim.run()
    sim.plot()
    sim.plot_metrics()
