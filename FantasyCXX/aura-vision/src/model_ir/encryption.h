#ifndef VISION_ENCRYPTION_H
#define VISION_ENCRYPTION_H

#ifdef USE_EXT_ENCRYPT
#include "wbaeslib.h"
#endif

namespace aura::vision {

    /**
     * @brief 模型加密类型
     * */
    enum ModelEncryptType {
        TP_NON = 0,
        TP_DES = 1,
        TP_AES = 2
    };

    /**
     * @brief DES 密钥数据
     * */
    struct DesKey {
        unsigned char k[8];
        unsigned char c[4];
        unsigned char d[4];
    };

    /**
     * @brief 加密、解密工具类
     * */
    class Encryption {
    public:
        Encryption();

        ~Encryption();

        void init(int model_encrypt_type, char* aes_encrypt_data, char* aes_decrypt_data);

        /**
         * @brief 数据加密
         * @param model_encrypt_type 模型文件加密的类型
         * @param src   输入原始的数据串
         * @param dst   输出加密后的数据串
         * @param len   数据的长度
         * @return 0-成功
         * */
        int encrypt(int model_encrypt_type, unsigned char *src, unsigned char *dst, int len);

        /**
         * @brief 数据解密
         * @param model_encrypt_type 模型文件加密的类型
         * @param src   输入原始的数据串
         * @param dst   输出加密后的数据串
         * @param len   数据的长度
         * @return 0-成功
         * */
        int decrypt(int model_encrypt_type, unsigned char *src, unsigned char *dst, int len);

    private:

        /**
         * @brief AES 数据加密
         * @param src   输入原始的数据串
         * @param dst   输出加密后的数据串
         * @param len   数据的长度
         * */
        int aes_encrypt(unsigned char *src, unsigned char *dst, int len);

        /**
         * @brief AES 数据解密
         * @param src   输入加密的数据串
         * @param dst   输出解密后的数据串
         * @param len   数据的长度
         * */
        int aes_decrypt(unsigned char *src, unsigned char *dst, int len);

        /**
         * @brief DES 数据加密
         * @param src   输入原始的数据串
         * @param dst   输出加密后的数据串
         * @param len   数据的长度
         * */
        int des_encrypt(unsigned char *src, unsigned char *dst, int len);

        /**
         * @brief DES 数据解密
         * @param src   输入加密的数据串
         * @param dst   输出解密后的数据串
         * @param len   数据的长度
         * */
        int des_decrypt(unsigned char *src, unsigned char *dst, int len);

        /**
         * @brief 生成 DES 密钥
         * */
        void generate_des_keys();

        /**
         * @brief 执行 DES 加解密计算
         * @param input_piece 
         * */
        void des_process_message(unsigned char *input_piece, unsigned char *output_piece, int mode_type);

    private:
        void *_m_aes_encrypt_ptr;
        void *_m_aes_decrypt_ptr;
        DesKey *_m_des_key;
    };
}

#endif //VISION_ENCRYPTION_H
