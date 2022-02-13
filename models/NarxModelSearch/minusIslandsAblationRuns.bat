cls
for /l %%x in (1, 1, 120) do (
    echo Run %%x

    REM Restart rabbitmq_server
    "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.8.9\sbin\rabbitmqctl.bat" stop_app
    "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.8.9\sbin\rabbitmqctl.bat" start_app

    REM Spawn worker
    start "worker" python WorkerModelTrainer.py -g 0

    REM Start islands

    if %%x leq 20 (
        mpiexec -n 21 python mpiNeuroevolutionIslands.py -e all_types
    ) else  (
        if %%x leq 40 (
            mpiexec -n 21 python mpiNeuroevolutionIslands.py -e noPSO
        ) else (
            if %%x leq 60 (
                mpiexec -n 21 python mpiNeuroevolutionIslands.py -e noGA
            ) else (
                if %%x leq 80 (
                    mpiexec -n 21 python mpiNeuroevolutionIslands.py -e noBO
                ) else (
                    if %%x leq 100 (
                        mpiexec -n 21 python mpiNeuroevolutionIslands.py -e noDE
                    ) else (
                        if %%x leq 120 (
                            mpiexec -n 21 python mpiNeuroevolutionIslands.py -e noRand
                        )
                    )
                )
            )
        )
    )
)
