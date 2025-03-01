import Button from "ui/button/Button";
import { Form, Input, Rate } from "antd";
import { IFormReviewProps, TFormValues } from "./types";
import s from "./styles.module.scss"

const { TextArea } = Input;

export const ReviewForm = ({ title = "Отзыв о месте" }: IFormReviewProps) => {
    const [form] = Form.useForm();

    const onFinish = (values: TFormValues) => {
        //some code
        form.resetFields();
    };

    return (
        <>
            <h2 className={s.title}>{title}</h2>
            <Form onFinish={onFinish} form={form}>
                <Form.Item name="rating" rules={[{ required: true, message: "Укажите рейтинг" }]}>
                    <Rate allowClear={false}/>
                </Form.Item>
                <Form.Item name="text" rules={[{ required: true, message: "Напишите отзыв" }]}>
                    <TextArea style={{resize: "none" }} rows={4}/>
                </Form.Item>
                <Button type="primary" htmlType="submit">Отправить</Button>
            </Form>
        </>
    );
}

