cls
for /l %%x in (1, 1, 420) do (
    echo Run %%x

    REM Restart rabbitmq_server
    "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.8.9\sbin\rabbitmqctl.bat" stop_app
    "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.8.9\sbin\rabbitmqctl.bat" start_app

    REM Spawn worker
    start "worker" python WorkerModelTrainer.py -g 0

    REM Start islands

    if %%x leq 30 (
        mpiexec -n 9 python mpiNeuroevolutionIslands.py -e all_types
    ) else  (
        if %%x leq 60 (
            mpiexec -n 9 python mpiNeuroevolutionIslands.py -e noPSO
        ) else (
            if %%x leq 90 (
                mpiexec -n 9 python mpiNeuroevolutionIslands.py -e noGA
            ) else (
                if %%x leq 120 (
                    mpiexec -n 9 python mpiNeuroevolutionIslands.py -e noBO
                ) else (
                    if %%x leq 150 (
                        mpiexec -n 9 python mpiNeuroevolutionIslands.py -e noDE
                    ) else (
                        if %%x leq 180 (
                            mpiexec -n 9 python mpiNeuroevolutionIslands.py -e noRand
                        ) else (
                            if %%x leq 210 (
                                mpiexec -n 9 python mpiNeuroevolutionIslands.py -e onlyPSO
                            ) else (
                                if %%x leq 240 (
                                    mpiexec -n 9 python mpiNeuroevolutionIslands.py -e onlyGA
                                ) else (
                                    if %%x leq 270 (
                                        mpiexec -n 9 python mpiNeuroevolutionIslands.py -e onlyBO
                                    ) else (
                                        if %%x leq 300 (
                                            mpiexec -n 9 python mpiNeuroevolutionIslands.py -e onlyDE
                                        ) else (
                                            if %%x leq 330 (
                                                mpiexec -n 9 python mpiNeuroevolutionIslands.py -e onlyRand
                                            ) else (
                                                if %%x leq 360 (
                                                    mpiexec -n 6 python mpiNeuroevolutionIslands.py -e onlyLS
                                                ) else (
                                                    if %%x leq 390 (
                                                        mpiexec -n 9 python mpiNeuroevolutionIslands.py -e 2_BO_PSO_RAND_1_GA_DE
                                                    ) else (
                                                        mpiexec -n 9 python mpiNeuroevolutionIslands.py -e 3_BO_2_PSO_1_RAND_GA_DE
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
)
