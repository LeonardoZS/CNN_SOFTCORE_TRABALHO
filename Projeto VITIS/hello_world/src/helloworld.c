#include <stdio.h>
#include "platform.h"
#include "xil_printf.h"
#include <stdint.h>

#include "cnn_weights.h"
#include "test_image.h" 



//A saida e o resultado da convolucao (5x5)
//A entrada e a imagem disposta na figura acima (7x7).
//O kernel possui tamanho (3x3)
//O peso da polarizacao e unitario (1)
void conv(float output[5][5], uint8_t input[7][7], float kernel[3][3], float bias) 
{
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {

            float acc = 0.0;

            for (int kernel_pos_x = 0; kernel_pos_x < 3; kernel_pos_x++) 
            {
                for (int kernel_pos_y = 0; kernel_pos_y < 3; kernel_pos_y++) 
                {
                    acc += input[i + kernel_pos_x][j + kernel_pos_y] * kernel[kernel_pos_x][kernel_pos_y];
                }
            }

            output[i][j] = acc + bias;
        }
    }
}


//entrada 5x5, saida da primeira camada
//saida 2x2, reduzida pela metade, como 5 e impar, optei por reduzir para 2 e nao para 3.
void maxpool2d_5_to_2(float output[2][2], float input[5][5]) {
    for (int i = 0; i < 2; i++) 
    {
        for (int j = 0; j < 2; j++) 
        {
            float max_values = input[i * 2][j * 2];
            if (input[i * 2][j * 2 + 1] > max_values)
            {
                max_values = input[i * 2][j * 2 + 1];
            }
            if (input[i * 2 + 1][j * 2] > max_values)
            {
                max_values = input[i * 2 + 1][j * 2];
            }
            if (input[i * 2 + 1][j * 2 + 1] > max_values) 
            {
                max_values = input[i * 2 + 1][j * 2 + 1];
            }
            output[i][j] = max_values;
        }
    }
}

//entrada 2x2
//saida 1x4
void flatten(float output[4], float input[2][2]) {
    int idx = 0;
    for (int i = 0; i < 2; i++) 
    {
        for (int j = 0; j < 2; j++) 
        {
            output[idx++] = input[i][j];
        }
    }
}

//entrada 1x4
//saida 10 [softmax]
void dense_result_final(float input[4], float output[10], float weights[4][10], float bias[10]) {
    for (int i = 0; i < 10; i++) 
    {
        output[i] = bias[i];

        for (int j = 0; j < 4; j++) 
        {
            output[i] += input[j] * weights[j][i];
        }
    }
}


int main() {
    init_platform();

    float conv_output[5][5];
    float pooled_min[2][2];
    float flatted[4];
    float output[10];

    uint8_t inf_result = 0;

    xil_printf("I");  //print para pegar o ínicio
    conv(conv_output, test_image, conv1_weights, conv1_bias);
    maxpool2d_5_to_2(pooled_min, conv_output);
    flatten(flatted, pooled_min);
    dense_result_final(flatted, output, dense_weights, dense_bias);
    xil_printf("F"); //print para pegar o fim

    for (int i = 0; i < 10; i++) 
    {
        if(output[i] > inf_result) inf_result = i;

    }

    //classse resultante
    xil_printf("%d", (int)(inf_result));

    cleanup_platform();
    return 0;
}
