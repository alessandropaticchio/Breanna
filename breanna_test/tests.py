def test_feature_extraction():
    from breanna import feature_extraction as fe
    from breanna import util
    from skimage import io
    
    import matplotlib.pyplot as plt
    
    test_image_path = 'test_data/frame1.jpg'
    test_image_path = util.to_ospath(test_image_path)             
    test_image = io.imread(test_image_path)
    plt.imshow(test_image)  
    plt.show()

    test_banner_path  = 'test_data/banner1'
    test_encoder_path = '../breanna_data/models/encoder/encoder64.h5'
    
    test_banner_root = '../breanna_data/banners/alfaromeo_march_banners/'
    
    extractor = fe.VisualFeatureExtractor(['GLC', 'DNB', 'GLS', 'DNC', 'DNE'])
    features = extractor.extract(test_image)
    assert features.shape      == (5,)
    assert extractor.get_dim() ==  5
    print('\nFeature Extraction using visual features returns: ')
    print(list(features))
 
    aggregator = fe.FeatureAggregator(extractor, method='mean')
    aggregated_features = aggregator.aggregate(test_banner_path)
    assert aggregated_features.shape == (5,)
    print('\nFeature Aggregated using visual features returns: ')
    print(list(aggregated_features))
    
    extractor = fe.AutoEncoderExtractor(test_encoder_path, (64, 64), 64)  
    features  = extractor.extract(test_image)
    assert features.shape      == (64,)
    assert extractor.get_dim() ==  64
    print('\nFeature Extraction using autoencoder returns: ')
    print(list(features))
 
    aggregator = fe.FeatureAggregator(extractor, method='mean')
    aggregated_features = aggregator.aggregate(test_banner_path)
    assert aggregated_features.shape == (64,)
    print('\nFeature Aggregated using autoencoder returns: ')
    print(list(aggregated_features))
       
    test_df = fe.summarize_banners(test_banner_root, aggregator)
    print(test_df.head())
    
def test_database_access():
    from breanna import database_access as da
    
    msl_db = da.get_connection() 
    da.initialize_database(conn=msl_db)

    da.load_new_campaign('automobile', 'alfaromeo', '3', '2019', 
                      banner_root=         '../breanna_data/alfaromeo_march/alfaromeo_march_banners',
                      standarddisplay_root='../breanna_data/alfaromeo_march/alfaromeo_march_standarddisplay',
                      matchfile_root=      '../breanna_data/alfaromeo_march/alfaromeo_march_matchfile',
                      conn=msl_db)

def test_ctr_model_management():
    from breanna import feature_extraction as fe
    from breanna import alfaromeo_specific as ar
    from breanna import database_access    as da
    from breanna import ctr_model_management as cmm
    from breanna import util
    
    from sklearn.ensemble import RandomForestRegressor
    from skimage import io
    
    extractor  = fe.VisualFeatureExtractor(['GLC', 'DNB', 'GLS', 'DNC', 'DNE'])
    aggregator = fe.FeatureAggregator(extractor, method='mean')

    ctr_aggregators = ['event_hour', 'publisher', 'device']
    X_onehot, y, banner_names = cmm.create_training_set(ctr_aggregators=ctr_aggregators, 
                                          ontology=['automobile'], name=['alfaromeo'], year=['2019'], month=['3'], 
                                          min_impressions=5000,
                                          viz_extractor=extractor, viz_aggregator=aggregator,
                                          matching_rule=ar.are_referring_same_banner,
                                          conn=da.get_connection())

    rf_regressor = RandomForestRegressor(n_estimators=100)
    
    cmm.evaluate_estimator(X_onehot, y, banner_names, rf_regressor)
    
    ctrmodel = cmm.train_CTRModel(X_onehot, y, rf_regressor,
                                  viz_extractor=extractor,
                                  ctr_aggregators=ctr_aggregators)

    cmm.save_CTRModel(ctrmodel,  'event_hour-publisher-device-randomforestregressor.p')
    ctrmodel = cmm.load_CTRModel('event_hour-publisher-device-randomforestregressor.p')
  
    test_image_path = 'test_data/frame1.jpg'
    test_image_path = util.to_ospath(test_image_path)             
    test_image = io.imread(test_image_path)

    print( ctrmodel.predict(test_image) )
    
if __name__ == '__main__':
#    test_feature_extraction()
#    test_database_access()
#    test_ctr_model_management()