package org.to0mi1.sample.cifar10evaluate.domain;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.to0mi1.sample.cifar10evaluate.ResultDto;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

@Service
public class DeepLearningService {

    private static final String[] LABELS = {
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"};

    @Autowired
    private MultiLayerNetwork multiLayerNetwork;

    public List<ResultDto> predict(InputStream is) {

        try {
            NativeImageLoader loader = new NativeImageLoader(32, 32, 3);

            INDArray image = loader.asMatrix(is);

            DataNormalization scalar = new ImagePreProcessingScaler(0, 1);

            scalar.transform(image);

            INDArray output = multiLayerNetwork.output(image);

            List<ResultDto> resultDtos = new ArrayList<>();
            for (int i = 0; i < output.columns(); i++) {
                resultDtos.add(new ResultDto(LABELS[i], output.getDouble(i)));
            }
            return resultDtos;

        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
