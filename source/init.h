    // INITIALIZE BROADCAST AND REDUCTION TREES
    void init(int numlevel, int groupsize[], CommBench::library lib[], int numstripe, int numbatch) {

      if(myid == printid) {
        printf("NUMBER OF EPOCHS: %d\n", numepoch);
        for(int epoch = 0; epoch < numepoch; epoch++)
          printf("epoch %d: %zu bcast %zu reduction\n", epoch, bcast_epoch[epoch].size(), reduce_epoch[epoch].size());
        printf("Initialize HiCCL with %d levels\n", numlevel);
        for(int level = 0; level < numlevel; level++) {
          printf("level %d groupsize %d library: ", level, groupsize[level]);
          CommBench::print_lib(lib[level]);
          if(level == 0)
            if(groupsize[0] != numproc)
              printf(" *");
          printf("\n");
        }
        printf("\n");
      }

      // ALLOCATE COMMAND BATCH
      for(int batch = 0; batch < numbatch; batch++)
        coll_batch.push_back(std::list<Coll<T>*>());

      // TEMP HIERARCHY FOR TREE
      std::vector<int> groupsize_temp(groupsize, groupsize + numlevel);
      groupsize_temp[0] = numproc;

      // FOR EACH EPOCH
      for(int epoch = 0; epoch < numepoch; epoch++) {
        // INIT BROADCAST
        std::vector<BROADCAST<T>> &bcastlist = bcast_epoch[epoch];
        if(bcastlist.size()) {
          // PARTITION INTO BATCHES
          std::vector<std::vector<BROADCAST<T>>> bcast_batch(numbatch);
          partition(bcastlist, numbatch, bcast_batch);
          // FOR EACH BATCH
          for(int batch = 0; batch < numbatch; batch++) {
            // STRIPE BROADCAST PRIMITIVES
            std::vector<REDUCE<T>> split_list;
            stripe(numstripe, bcast_batch[batch], split_list);

            // APPLY REDUCE TREE TO ROOTS FOR STRIPING
            std::vector<T*> recvbuff; // for memory recycling
            // reduce_tree(numlevel, groupsize_temp.data(), lib, split_list, numlevel - 1, coll_batch[batch], recvbuff, 0);
            reduce_tree(1, groupsize_temp.data(), &lib[numlevel-1], split_list, 0, coll_batch[batch], recvbuff, 0);

            // APPLY RING TO BRANCHES ACROSS NODES
            std::vector<BROADCAST<T>> bcast_intra; // for accumulating intra-node communications for tree (internally)
            bcast_ring(groupsize[0], lib[0], bcast_batch[batch], bcast_intra, coll_batch[batch]);

            // APPLY TREE TO THE LEAVES WITHIN NODES
            bcast_tree(numlevel, groupsize_temp.data(), lib, bcast_intra, 1, coll_batch[batch]);
          }
        }
        // INIT REDUCTION
        std::vector<REDUCE<T>> &reducelist = reduce_epoch[epoch];
        if(reducelist.size()) {
          // PARTITION INTO BATCHES
          std::vector<std::vector<REDUCE<T>>> reduce_batch(numbatch);
          partition(reducelist, numbatch, reduce_batch);
          // FOR EACH BATCH
          for(int batch = 0; batch < numbatch; batch++) {
            // STRIPE REDUCTION
            std::vector<BROADCAST<T>> merge_list;
            stripe(numstripe, reduce_batch[batch], merge_list);
            // HIERARCHICAL REDUCTION RING + TREE
            std::vector<REDUCE<T>> reduce_intra; // for accumulating intra-node communications for tree (internally)
            reduce_ring(numlevel, groupsize, lib, reduce_batch[batch], reduce_intra, coll_batch[batch]);
            // COMPLETE STRIPING BY INTRA-NODE GATHER
            bcast_tree(numlevel, groupsize_temp.data(), lib, merge_list, 1, coll_batch[batch]);
          }
        }
      }
      // IMPLEMENT WITH COMMBENCH
      implement(coll_batch, command_batch, 1);
    }


