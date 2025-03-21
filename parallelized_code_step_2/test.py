from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = 1000
else:
    data = None  # Les autres processus n'ont rien au début

# Diffusion depuis le processus 0 vers tous les autres
data = comm.bcast(data, root=0)

print(f"Process {rank} received data: {data}")
