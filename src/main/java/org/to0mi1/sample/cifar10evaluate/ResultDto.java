package org.to0mi1.sample.cifar10evaluate;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ResultDto {
    String label;
    double confidence;
}
