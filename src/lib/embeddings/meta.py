import lib.embeddings as embeddings


def get_embedder(args):
    if args.MODEL.embedding_flag:
        if args.MODEL.embedding_algorithm == "lle":
            trn_embedder = embeddings.locally_linear_embedding.LLE(
                n_neighbors=args.TRAIN.lle_n_neighbors,
                n_class=args.TRAIN.n_way,
                n_support=args.TRAIN.n_support,
                n_query=args.TRAIN.n_query,
                device=args.device
            )
            val_embedder = embeddings.locally_linear_embedding.LLE(
                n_neighbors=args.TEST.lle_n_neighbors,
                n_class=args.TEST.n_way,
                n_support=args.TEST.n_support,
                n_query=args.TEST.n_query,
                device=args.device
            )

        elif args.MODEL.embedding_algorithm == "mds":
            trn_embedder = embeddings.mds.MDSTorch(
                metric_type=args.MODEL.mds_metric_type
            )
            val_embedder = trn_embedder

        elif args.MODEL.embedding_algorithm == "svd":
            trn_embedder = embeddings.svd.SVDTorch()
            val_embedder = trn_embedder

        elif args.MODEL.embedding_algorithm == "lpp":
            trn_embedder = embeddings.lpp_robust.LocalityPreservingProjection()
            val_embedder = trn_embedder

        elif args.MODEL.embedding_algorithm == "lda":
            trn_embedder = embeddings.lda.lda_for_episode
            val_embedder = embeddings.lda.lda_for_episode

        elif args.MODEL.embedding_algorithm == "nca":
            trn_embedder = embeddings.nca.NCATrainer(
                input_dim=args.MODEL.output_dim,
                output_dim=args.MODEL.embedding_n_components,
                max_batch_size=args.DATA.batch_size*2,
                init_method=args.MODEL.init_nca_method,
                distance_method=args.MODEL.mds_metric_type,
                scale=args.MODEL.nca_scale,
                is_instanciate_each_iter=args.MODEL.is_instanciate_each_iter,
                device=args.device
                )
            val_embedder = embeddings.nca.NCATrainer(
                input_dim=args.MODEL.output_dim, output_dim=args.MODEL.embedding_n_components, is_instanciate_each_iter=args.MODEL.is_instanciate_each_iter)

        else:
            raise NotImplementedError(
                f"embedding algotirhm \'{args.MODEL.embedding_algorithm}\' not implemented")
    else:
        trn_embedder = None
        val_embedder = None

    return trn_embedder, val_embedder
