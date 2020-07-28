# TODO

* understand heuristic / clean up / document - done
* run heuristic controller - done
* create ai controller - done
* allow multiple controller - done
* add AI to environment - done
* make video - done
* parameterize h agents - done
* parameterize rewards - done
* run with djx - done
* make plots - done

* make tabularq working with history - done
* save table as csv - done
* make ci, ai optional - done
* allow for fixed network (fixed position, fixed network) - done
* think of usefull networks to test - done
* reduce filesize - done
    <!-- * test other storage format -->
    * save as parquet
    * save individual agents less often
* run grid - done
* think of usefull plot to make - done
* create plots - done



* update writing
    * different files for different type of logs - done
    * seperate meta and data - not needed
    * log all metrics individual, then mix for final reward - done
    * sum reward per group (AI / CI) - done
    * CI reward - ???
* check qtables simple settings, is it working correctly?




to test:
* topology
    * fully connected 2,3,4,5
    * ring degree 2, 4
* all agents
* fixed position, fixed network
* rewards
    * only agents
        * local only
        * local / global
        * global only
    * ai / ci
        * ci mixed, ai global
* controller
    * heuristic
        * self_weight [-0.5, 0, 0.5, 1, 2]
    * qtable (test with three other settings only)
        * alpha: [0.01, 0.05, 0.1]
        * gamma: [0.999, 0.99, 0.9, 0.5]
        * q_start: [-2, 0, 2]
        * obs_map: [product, combinations]
        * cache_size: [1,2]


test 1:
* topology
    * fully connected 2,3,4,5
    * ring degree 2, 4
* all agents
* fixed position, fixed network   
* rewards 
    * ai / ci
        * ci mixed, ai global
* ci_controller
    * heuristic
        * self_weight [-0.5, 0, 0.5, 1, 2]
* ai_controller
    * heuristic

test 2:
* topology
    * fully connected 3,5
    * ring degree 2, 4
* all agents
* fixed position, fixed network
* rewards
    * only agents
        * local only
        * local / global
        * global only
    * ai / ci
        * ci mixed, ai global
* controller
    * qtable (test with three other settings only)
        * alpha: [0.01, 0.05, 0.1]
        * gamma: [0.999, 0.99, 0.9, 0.5]
        * q_start: [-2, 0, 2]
        * obs_map: [product, combinations]
        * cache_size: [1,2]



test 3:
* topology
    * fully connected 2,3,4,5
    * ring degree 2, 4
* all agents
* fixed position, fixed network   
* rewards
    * only agents
        * local only
        * local / global
        * global only
    * ai / ci
        * ci mixed, ai global
* ci_controller
    * heuristic
        * self_weight [-0.5, 0, 0.5, 1, 2]
* ai_controller
    * qtable
        * alpha: 0.05
        * gamma: 0.999
        * q_start: 0
        * obs_map: product
        * cache_size: [1,2]