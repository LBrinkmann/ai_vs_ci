dvc run -f runs/djx_test/num_episodes_100__n_agents_5__networktype_nondense/dvc.dvc -d aci/multi_train.py -d runs/djx_test/num_episodes_100__n_agents_5__networktype_nondense/params.yml \
    -o runs/djx_test/num_episodes_100__n_agents_5__networktype_nondense/data \
    python aci/multi_train.py runs/djx_test/num_episodes_100__n_agents_5__networktype_nondense/params.yml runs/djx_test/num_episodes_100__n_agents_5__networktype_nondense/data > runs/djx_test/num_episodes_100__n_agents_5__networktype_nondense/log.log