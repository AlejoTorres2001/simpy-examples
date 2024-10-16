import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

THINKING_MEAN = 25
SERVICE_MEAN = 0.8
QUANTUM = 0.1
CONTEXT_SWITCH = 0.015
TOTAL_JOBS = 1000

def exponential(mean):
    return random.expovariate(1.0 / mean)

class Terminal:
    def __init__(self, env, cpu, results, wait_times):
        self.env = env
        self.cpu = cpu
        self.results = results
        self.wait_times = wait_times
        self.thinking_time = exponential(THINKING_MEAN)
        self.process = env.process(self.run())
        
    def run(self):
        while len(self.results) < TOTAL_JOBS:
            yield self.env.timeout(self.thinking_time)
            
            service_time = exponential(SERVICE_MEAN)
            arrival_time = self.env.now
            
            start_wait = self.env.now
            yield self.env.process(self.cpu.process_job(service_time))
            wait_time = self.env.now - start_wait
            
            response_time = self.env.now - arrival_time
            self.results.append(response_time)
            self.wait_times.append(wait_time)
            
            self.thinking_time = exponential(THINKING_MEAN)

class CPU:
    def __init__(self, env):
        self.env = env
        self.queue = simpy.Store(env)
        self.working = False
        self.busy_time = 0  # Tiempo total de CPU ocupada
        self.queue_lengths = []  # Registrar longitudes de cola
        
    def process_job(self, service_time):
        arrival_time = self.env.now
        yield self.queue.put(service_time)
        if not self.working:
            self.working = True
            yield self.env.process(self.start_processing())
            
    def start_processing(self):
        while len(self.queue.items) > 0:
            remaining_time = yield self.queue.get()
            
            while remaining_time > 0:
                quantum_time = min(remaining_time, QUANTUM)
                yield self.env.timeout(quantum_time + CONTEXT_SWITCH)
                self.busy_time += quantum_time + CONTEXT_SWITCH
                remaining_time -= quantum_time
                
            self.queue_lengths.append(len(self.queue.items))
        self.working = False

def simulate_system(n_terminals):
    env = simpy.Environment()
    cpu = CPU(env)
    results = []
    wait_times = []
    
    for _ in range(n_terminals):
        Terminal(env, cpu, results, wait_times)
    
    env.run()
    
    mean_response_time = np.mean(results)
    mean_wait_time = np.mean(wait_times)
    cpu_utilization = cpu.busy_time / env.now
    mean_queue_length = np.mean(cpu.queue_lengths)
    
    return mean_response_time, mean_wait_time, cpu_utilization, mean_queue_length

# Almacenar métricas
response_times = []
wait_times = []
cpu_utilizations = []
queue_lengths = []
terminal_counts = [5, 10, 20, 40, 80]

# Ejecutar simulaciones
for n in terminal_counts:
    mean_response_time, mean_wait_time, cpu_utilization, mean_queue_length = simulate_system(n)
    response_times.append(mean_response_time)
    wait_times.append(mean_wait_time)
    cpu_utilizations.append(cpu_utilization)
    queue_lengths.append(mean_queue_length)

# Graficar resultados
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(terminal_counts, response_times, marker='o')
plt.title('Tiempo medio de respuesta vs Número de terminales')
plt.xlabel('Número de terminales')
plt.ylabel('Tiempo medio de respuesta (s)')

plt.subplot(2, 2, 2)
plt.plot(terminal_counts, wait_times, marker='o')
plt.title('Tiempo medio de espera en cola vs Número de terminales')
plt.xlabel('Número de terminales')
plt.ylabel('Tiempo medio de espera (s)')

plt.subplot(2, 2, 3)
plt.plot(terminal_counts, cpu_utilizations, marker='o')
plt.title('Utilización de la CPU vs Número de terminales')
plt.xlabel('Número de terminales')
plt.ylabel('Utilización de la CPU (%)')

plt.subplot(2, 2, 4)
plt.plot(terminal_counts, queue_lengths, marker='o')
plt.title('Longitud media de la cola vs Número de terminales')
plt.xlabel('Número de terminales')
plt.ylabel('Longitud media de la cola')

plt.tight_layout()
plt.show()