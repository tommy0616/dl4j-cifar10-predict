package org.to0mi1.sample.cifar10evaluate;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;

import java.io.IOException;

@Configuration
public class DLConfig {

    @Autowired
    private ResourceLoader resourceLoader;

    @Bean
    public MultiLayerNetwork multiLayerNetwork() {
        Resource weightResource = resourceLoader.getResource("classpath:weights.078-0.47.hdf5");
        try {
            String simpleMlp = weightResource.getFile().getPath();
            return KerasModelImport.importKerasSequentialModelAndWeights(simpleMlp);
        } catch (IOException | InvalidKerasConfigurationException | UnsupportedKerasConfigurationException e) {
            throw new RuntimeException(e);
        }
    }
}
