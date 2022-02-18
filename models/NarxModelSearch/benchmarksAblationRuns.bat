cls
for /l %%x in (101, 1, 140) do (
    echo Run %%x

    REM Restart rabbitmq_server
    "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.8.9\sbin\rabbitmqctl.bat" stop_app
    "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.8.9\sbin\rabbitmqctl.bat" start_app

    REM Spawn worker
    start "worker" python WorkerModelTrainer.py -g 0

    REM Start islands

    if %%x leq 20 (
        REM mpiexec -n 6 python mpiNeuroevolutionIslands.py -e onlyRand
    ) else  (
        if %%x leq 40 (
            mpiexec -n 5 python mpiNeuroevolutionIslands.py -e all_types
        ) else (
            if %%x leq 60 (
                mpiexec -n 9 python mpiNeuroevolutionIslands.py -e all_types
            ) else (
                if %%x leq 80 (
                    mpiexec -n 17 python mpiNeuroevolutionIslands.py -e all_types
                ) else (
                    if %%x leq 100 (
                        mpiexec -n 25 python mpiNeuroevolutionIslands.py -e all_types
                    ) else (
                        if %%x leq 120 (
                            mpiexec -n 33 python mpiNeuroevolutionIslands.py -e all_types
                        ) else (
                            if %%x leq 140 (
                                mpiexec -n 41 python mpiNeuroevolutionIslands.py -e all_types
                            )
                        )
                    )
                )
            )
        )
    )
)
