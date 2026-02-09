import { View, Text, Pressable, StyleSheet } from 'react-native';
import { Link, Href } from 'expo-router';

type NavButtonProps = {
    label: string;
    href: Href;
};

export default function NavButton({ label, href }: NavButtonProps) {
    return (
        // As child lets link pass navigation behavior into pressable, instead of wrapping it a web like anchor
        <Link href={href} asChild>
            {/* 'pressed' is passed to function which returns an array, only gives 'styles.pressed' when 'pressed' is true. It will override earlier array objects*/}
            {/* <Pressable style={({ pressed }) => [styles.button, pressed && styles.pressed]}> */}
            <Pressable android_ripple={{ color: '#ccc' }} style={styles.button}>
                <Text style={styles.text}>{label}</Text>
            </Pressable>
        </Link>
    );
}

const styles = StyleSheet.create({
    button: {
        padding: 14,
        backgroundColor: '#1162ccff',
        borderRadius: 8,
        alignItems: 'center',
    },
    pressed: {
        backgroundColor: '#aa0e30ff',
    },
    text: {
        fontSize: 16,
        fontWeight: '600',
    },
});
