package org.to0mi1.sample.cifar10evaluate.domain;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ResourceLoader;
import org.springframework.test.context.junit4.SpringRunner;
import org.to0mi1.sample.cifar10evaluate.ResultDto;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.List;

@SpringBootTest
@RunWith(SpringRunner.class)
//@AutoConfigureMockMvc
public class DeepLearningServiceTest {

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
    DeepLearningService dlService;

    @Autowired
    private ResourceLoader resourceLoader;

    @Test
    public void testEvaluate() throws IOException {
        InputStream is = resourceLoader.getResource("classpath:download.jpg").getInputStream();

        List<ResultDto> predict = dlService.predict(is);

        DecimalFormat decimalFormat = new DecimalFormat("0.##");
        for (ResultDto dto : predict) {
            System.out.println(dto.getLabel() + " : " + decimalFormat.format(dto.getConfidence()));
        }

    }


}