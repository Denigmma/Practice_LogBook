# Исходные данные
cpu_burst = {"p1": 41, "p2": 52}
io_burst = {"p1": 73, "p2": 84}
quantum = 20
switch_time = 5

# Состояние процессов
remaining_cpu = {"p1": cpu_burst["p1"], "p2": cpu_burst["p2"]}
remaining_io = {"p1": 0, "p2": 0}
io_active = {"p1": False, "p2": False}
# Результаты
cpu_time = {"p1": 0, "p2": 0}
# Время
current_time = 0
current_process = "p1"

while current_time < 500:
    if remaining_cpu[current_process] > 0:
        executed_time = min(quantum, remaining_cpu[current_process])
        cpu_time[current_process] += executed_time
        current_time += executed_time
        remaining_cpu[current_process] -= executed_time

        if remaining_cpu[current_process] == 0:
            io_active[current_process] = True
            remaining_io[current_process] = io_burst[current_process]

    current_time += switch_time
    current_process = "p2" if current_process == "p1" else "p1"

    for proc in ["p1", "p2"]:
        if io_active[proc]:
            remaining_io[proc] -= switch_time
            if remaining_io[proc] <= 0:
                io_active[proc] = False
                remaining_cpu[proc] = cpu_burst[proc]

print(cpu_time["p1"], cpu_time["p2"])
