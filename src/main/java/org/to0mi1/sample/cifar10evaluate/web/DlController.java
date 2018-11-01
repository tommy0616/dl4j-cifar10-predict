package org.to0mi1.sample.cifar10evaluate.web;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;
import org.to0mi1.sample.cifar10evaluate.ResultDto;
import org.to0mi1.sample.cifar10evaluate.domain.DeepLearningService;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;


@Controller
public class DlController {

    @Autowired
    private DeepLearningService deepLearningService;

    @RequestMapping(value = "/", method = RequestMethod.GET)
    public String showHome(Model model) {
        return "home";
    }

    @RequestMapping(value = "/", method = RequestMethod.POST)
    public String evaluateImage(@RequestParam(name = "predictFormImage") MultipartFile predictMultipartFile,
                                Model model) {
        try {
            InputStream is = predictMultipartFile.getInputStream();
            List<ResultDto> resultDtos = deepLearningService.predict(is);
            model.addAttribute("results", resultDtos);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return "home";
    }
}
